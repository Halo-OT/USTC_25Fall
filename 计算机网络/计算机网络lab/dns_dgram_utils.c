#include <arpa/inet.h>
#include <assert.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include "dns_relay.h"
#include "dns_dgram_utils.h"

/*
    parse the domain name, type and class from question section of a dns datagram
    input:
        buf: the pointer point to the begin of the dns datagram
    output:
        name: the resolved domain name
        question: other fields except domain name in question section
    note:
        - support both sequences of labels and pointer
        - in this lab, consider that the dns request datagram contains ONLY one entry in question section for convenience.
*/
void parse_question_section(char *name, dns_question_t *question, const unsigned char *buf) {
    /* 
        TODO: implement this function 
    */
    const unsigned char *ptr = buf + DNS_HEADER_SIZE;
    char *name_ptr = name;
    
    // Parse domain name
    while (*ptr != 0) {
        // Check if it's a pointer (compression)
        if ((*ptr & 0xC0) == 0xC0) {
            // Pointer: 2 bytes, high 2 bits are 11
            uint16_t offset = ntohs(*(uint16_t *)ptr) & 0x3FFF;
            const unsigned char *jump_ptr = buf + offset;
            
            // Follow the pointer recursively
            while (*jump_ptr != 0) {
                if ((*jump_ptr & 0xC0) == 0xC0) {
                    uint16_t new_offset = ntohs(*(uint16_t *)jump_ptr) & 0x3FFF;
                    jump_ptr = buf + new_offset;
                } else {
                    // Label
                    uint8_t label_len = *jump_ptr++;
                    for (int i = 0; i < label_len; i++) {
                        *name_ptr++ = *jump_ptr++;
                    }
                    if (*jump_ptr != 0) {
                        *name_ptr++ = '.';
                    }
                }
            }
            ptr += 2;
            break;
        } else {
            // Label: first byte is length
            uint8_t label_len = *ptr++;
            for (int i = 0; i < label_len; i++) {
                *name_ptr++ = *ptr++;
            }
            if (*ptr != 0) {
                *name_ptr++ = '.';
            }
        }
    }
    
    *name_ptr = '\0';
    
    // Skip the null terminator if not already skipped by pointer
    if (*ptr == 0) {
        ptr++;
    }
    
    // Parse question type and class
    question->type = ntohs(*(uint16_t *)ptr);
    ptr += 2;
    question->cls = ntohs(*(uint16_t *)ptr);
    
    return;
}

/**
    try to find answer to the domain name by reading the local host file
    input:
        name: the domain name try to answer
        question: other fields except domain name in question section
        file_path: the path to the local host file
    output:
        ip: the IP of multiple resource records in string format (eg. "192.168.1.1")
    return:
        0 if no record, positive if any record
    note: supports one IP mapping to multiple domain names per line
*/
int try_answer_local(char ip[MAX_ANSWER_COUNT][MAX_IP_BUFFER_SIZE], const char *name, const char *file_path) {
    /* 
        TODO: implement this function 
    */
    FILE *fp = fopen(file_path, "r");
    if (fp == NULL) {
        return 0;
    }
    
    char line[MAX_ENTRY_BUFFER_SIZE];
    int count = 0;
    
    while (fgets(line, sizeof(line), fp) != NULL && count < MAX_ANSWER_COUNT) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        
        // Skip empty lines (only whitespace)
        int is_empty = 1;
        for (int j = 0; line[j] != '\0'; j++) {
            if (line[j] != ' ' && line[j] != '\t' && line[j] != '\n' && line[j] != '\r') {
                is_empty = 0;
                break;
            }
        }
        if (is_empty) continue;
        
        // Parse line: IP domain1 domain2 ...
        char ip_addr[MAX_IP_BUFFER_SIZE];
        char *ptr = line;
        
        // Skip leading whitespace
        while (*ptr == ' ' || *ptr == '\t') ptr++;
        
        // Read IP address
        int i = 0;
        while (*ptr != ' ' && *ptr != '\t' && *ptr != '\n' && *ptr != '\r' && *ptr != '\0') {
            ip_addr[i++] = *ptr++;
        }
        ip_addr[i] = '\0';
        
        // Check each domain name on this line
        while (*ptr != '\n' && *ptr != '\r' && *ptr != '\0') {
            // Skip whitespace
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            
            if (*ptr == '\n' || *ptr == '\r' || *ptr == '\0') break;
            
            // Read domain name
            char domain[MAX_DOMAIN_NAME_BUFFER_SIZE];
            i = 0;
            while (*ptr != ' ' && *ptr != '\t' && *ptr != '\n' && *ptr != '\r' && *ptr != '\0') {
                domain[i++] = *ptr++;
            }
            domain[i] = '\0';
            
            // Compare with the queried name
            if (strcmp(domain, name) == 0) {
                // Found a match, add to result
                strcpy(ip[count], ip_addr);
                count++;
                break;  // Found match on this line, move to next line
            }
        }
    }
    
    fclose(fp);
    return count;
}

/**
    it's more convenient to transform a dns request datagram to a dns response datagram than to construct a new dns response datagram
    input:
        buf: original dns request datagram
        len: original dns request datagram length
        ip: the IP of multiple resource records in string format (eg. "192.168.1.1")
        count: how many IP bind to this domain name
        question: other fields except domain name in question section
    output:
        buf: new dns response datagram
    return:
        length of the new dns response datagram
    note: 
        - do not need domain name, use pointer instead
        - need to support both IPv4 and IPv6
 */
int transform_to_response(unsigned char *buf, int len, const char ip[MAX_ANSWER_COUNT][MAX_IP_BUFFER_SIZE], int count, const dns_question_t *question) {
    /* 
        TODO: implement this function 
    */
    dns_header_t *header = (dns_header_t *)buf;
    
    // Modify header to make it a response
    header->qr = 1;        // This is a response
    header->aa = 0;        // Not authoritative
    header->ra = 1;        // Recursion available
    header->rcode = 0;     // No error
    
    // Position pointer after the question section
    unsigned char *ptr = buf + len;
    int actual_count = 0;  // Track actually added answers
    
    // Add answer records for each IP
    for (int i = 0; i < count; i++) {
        // Determine if IPv4 or IPv6
        int is_ipv6 = (strchr(ip[i], ':') != NULL);
        
        if (!is_ipv6 && question->type == DNS_TYPE_A) {
            // IPv4 answer
            dns_answer_v4_t *answer = (dns_answer_v4_t *)ptr;
            
            // Name: pointer to the domain name in question section
            answer->name = htons(0xC00C);  // Pointer to offset 12 (after header)
            answer->type = htons(DNS_TYPE_A);
            answer->cls = htons(DNS_CLASS_IN);
            answer->ttl = htonl(DNS_DEFAULT_TTL);
            answer->len = htons(4);  // IPv4 address length
            
            // Convert IP string to binary
            struct in_addr addr;
            inet_pton(AF_INET, ip[i], &addr);
            answer->ip = addr.s_addr;
            
            ptr += DNS_ANSWER_v4_SIZE;
            len += DNS_ANSWER_v4_SIZE;
            actual_count++;
            
        } else if (is_ipv6 && question->type == DNS_TYPE_AAAA) {
            // IPv6 answer
            dns_answer_v6_t *answer = (dns_answer_v6_t *)ptr;
            
            // Name: pointer to the domain name in question section
            answer->name = htons(0xC00C);  // Pointer to offset 12 (after header)
            answer->type = htons(DNS_TYPE_AAAA);
            answer->cls = htons(DNS_CLASS_IN);
            answer->ttl = htonl(DNS_DEFAULT_TTL);
            answer->len = htons(16);  // IPv6 address length
            
            // Convert IP string to binary
            struct in6_addr addr;
            inet_pton(AF_INET6, ip[i], &addr);
            
            // Copy IPv6 address (16 bytes)
            memcpy(&answer->iph, &addr.s6_addr[0], 8);
            memcpy(&answer->ipl, &addr.s6_addr[8], 8);
            
            ptr += DNS_ANSWER_v6_SIZE;
            len += DNS_ANSWER_v6_SIZE;
            actual_count++;
        }
    }
    
    // Update answer count with actual number added
    header->ancount = htons(actual_count);
    
    return len;
}