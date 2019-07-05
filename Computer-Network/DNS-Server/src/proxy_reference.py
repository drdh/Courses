import threading
import socket
import socketserver

import dnslib
from gevent import event

class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        request_data = self.request[0]
        # 将请求转发到 外部 DNS
        redirect_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        redirect_socket.sendto(request_data, ('202.38.64.56', 53))
        response_data, address = redirect_socket.recvfrom(1024)


        q=dnslib.DNSRecord.parse(response_data)
        qname=q.q.qname
        print(qname)
        # 将114响应响应给客户
        client_socket = self.request[1]
        client_socket.sendto(response_data, self.client_address)


class Server(socketserver.ThreadingMixIn, socketserver.UDPServer):
    pass


if __name__ == "__main__":
    # 一下ip需换成自己电脑的ip
    
    server = Server(('114.214.176.57', 53), Handler)
    with server:
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        print('The DNS server is running at 172.16.42.254...')
        server_thread.join()