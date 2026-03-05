#include <arpa/inet.h>
#include <assert.h>
#include <netinet/in.h>
#include <string.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include "dns_relay.h"
#include "dns_relay_utils.h"
#include "dns_dgram_utils.h"

static int listen_fd, upstream_fd;
static client_info_t upstream_info;

static unsigned char buf[MAX_DATAGRAM_BUFFER_SIZE];
static struct pollfd poll_fds[2];  // 0: listen_fd, 1: upstream_fd
static client_info_t pending_clients[UINT16_MAX];

static void initialize() {
    struct sockaddr_in listen_addr, upstream_addr;

    memset(&listen_addr, 0, sizeof(listen_addr));
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_addr.s_addr = inet_addr(LISTEN_ADDR);
    listen_addr.sin_port = htons(LISTEN_PORT);

    memset(&upstream_addr, 0, sizeof(upstream_addr));
    upstream_addr.sin_family = AF_INET;
    upstream_addr.sin_addr.s_addr = inet_addr(UPSTREAM_ADDR);
    upstream_addr.sin_port = htons(UPSTREAM_PORT);

    upstream_info.addr = upstream_addr;
    upstream_info.len = sizeof(upstream_addr);

    int ret;

    listen_fd = socket(PF_INET, SOCK_DGRAM, 0);
    if (listen_fd < 0) handle_fatal("socket() error");

    ret = bind(listen_fd, (struct sockaddr*)&listen_addr, sizeof(listen_addr));
    if (ret < 0) handle_fatal("bind() error");

    upstream_fd = socket(PF_INET, SOCK_DGRAM, 0);
    if (upstream_fd < 0) handle_fatal("socket() error");

    // Setup poll file descriptors
    poll_fds[0].fd = listen_fd;
    poll_fds[0].events = POLLIN;
    poll_fds[1].fd = upstream_fd;
    poll_fds[1].events = POLLIN;
}

static void finalize() {
    close(listen_fd);
    close(upstream_fd);
}

int main() {
    // constraints to struct size
    assert(DNS_HEADER_SIZE == 12); // dns_header must be 12 bytes
    assert(DNS_QUESTION_SIZE == 4); // dns question except domain name must be 4 bytes
    assert(DNS_ANSWER_v4_SIZE == 16); // dns answer must be 16 bytes
    assert(DNS_ANSWER_v6_SIZE == 28); // dns answer must be 16 bytes

    initialize();

    while (1) {
        const int event_count = poll(poll_fds, 2, -1);
        if (event_count < 0) handle_fatal("poll() error");

        for (int i = 0; i < 2; i++) {
            if (!(poll_fds[i].revents & POLLIN)) {
                continue;
            }

            // extract event fd
            const int event_fd = poll_fds[i].fd;

            // receive dns datagram and collect client info
            client_info_t client_info;
            memset(&client_info, 0, sizeof(client_info_t));
            client_info.len = sizeof(client_info.addr);
            int len = recvfrom(event_fd, buf, MAX_DATAGRAM_BUFFER_SIZE, 0, (struct sockaddr *)&client_info.addr, &client_info.len);
            if (len < 0) handle_fatal("recvfrom() error");

            // parse dns_header
            dns_header_t *dns_header = (dns_header_t *) buf;

            if (event_fd != listen_fd) {
                // handle upstream
                send_datagram(listen_fd, buf, len, pending_clients[dns_header->id]);
                continue;
            }

            // handle listen
            // if a dns response datagram, then do nothing
            if (dns_header->qr) {
                continue;
            }

            char name[MAX_DOMAIN_NAME_BUFFER_SIZE];
            dns_question_t question;
            char ip[MAX_ANSWER_COUNT][MAX_IP_BUFFER_SIZE];
            /*
                TODO: implement the logic about how to handle datagrams from listen socket
                Note: 
                    1. before finishing this part, make sure that you have finished other TODO parts
                    2. functions in dns_dgram_utils.c and dns_relay_utils.h may help a lot
                    3. remember to call log_result() after sending a datagram
            */
            
            // Parse the question section to extract domain name and query info
            parse_question_section(name, &question, buf);
            
            // Try to answer from local mappings file
            int ip_count = try_answer_local(ip, name, HOST_FILE_PATH);
            
            if (ip_count > 0) {
                // Found in local mappings, construct and send response
                int new_len = transform_to_response(buf, len, ip, ip_count, &question);
                send_datagram(listen_fd, buf, new_len, client_info);
                log_result("local resolve", name);
            } else {
                // Not found locally, forward to upstream DNS server
                // Save client info for later response
                pending_clients[dns_header->id] = client_info;
                
                // Forward the query to upstream DNS server
                send_datagram(upstream_fd, buf, len, upstream_info);
                log_result("relay", name);
            }
        }
    }

    finalize();

    return 0;
}
