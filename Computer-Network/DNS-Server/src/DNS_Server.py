import threading
import socket
import socketserver
from dnslib import DNSRecord,QTYPE
import dnslib
import argparse
import time

localIP="222.195.84.207"
#DNS_server="202.38.64.56"
DNS_server="202.106.0.20"
ip_file="./dnsrelay.txt"

debug_print=False
debug_print2=False
count=0

class Handler(socketserver.BaseRequestHandler):    
    def handle(self): 
        global count       
        count=count+1 #序号
        request_data = self.request[0]
        client_socket = self.request[1]
        #内部搜索
        d=DNSRecord.parse(request_data)
        qname=str(d.q.qname)
        qid=d.header.id
        search=cache.get(qname)
        #print(qname)
        if search:
            ret=d.reply()
            # 不良网站
            if search=="0.0.0.0":
                ret.add_answer(dnslib.RR(qname,QTYPE.TXT,rdata=dnslib.TXT(warning)))
            else:
                ret.add_answer(dnslib.RR(qname,rdata=dnslib.A(search)))
            ret.header.id=qid
            if debug_print:
                print(time.asctime( time.localtime(time.time())),"  ",
                count,"  ",self.client_address,qname)
            elif debug_print2:
                print("\n\n\n")
                print("*******Request Data***********")
                print(d)
                print("********Client Address********")
                print(self.client_address)
                print("********Search Name***********")
                print(qname)
                print("********Search IP*************")
                print(search)

            client_socket.sendto(bytes(ret.pack()), self.client_address)
        else:
            # 将请求转发到 外部 DNS
            redirect_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            redirect_socket.sendto(request_data, (DNS_server, 53))
            response_data, address = redirect_socket.recvfrom(8192)

            if debug_print:
                print(time.asctime( time.localtime(time.time())),"  ",
                count,"  ",self.client_address,qname)
            elif debug_print2:
                print("\n\n\n")
                print("*******Request Data***********")
                print(d)
                print("********Client Address********")
                print(self.client_address)
                print("********Search Name***********")
                print(qname)
                print("********Search IP*************")
                print(DNSRecord.parse(response_data))
            # 将外部响应响应给客户
            client_socket.sendto(response_data, self.client_address)

    
#class Server(socketserver.ThreadingMixIn, socketserver.UDPServer):
#    pass


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("level",help="choose a debug level:1,2,3",type=int)
    parser.add_argument("--addr",help="DNS Server ip Addr",type=str)
    parser.add_argument("--ipfile",help="dns table, e.g. ip.txt",type=str)

    args=parser.parse_args()
    

    if args.level == 1:
        pass
    elif args.level == 2:
        debug_print=True
        if args.addr:
            DNS_server=args.addr
        if args.ipfile:
            ip_file=args.ipfile
    elif args.level == 3:
        debug_print2=True
        if args.addr:
            DNS_server=args.addr

    print("Debug level: ",args.level)
    print("DNS Server: ",DNS_server)
    print("IP table: ",ip_file)

    # 添加外部的ip地址
    f=open(ip_file,'r')
    ip=[]
    for a in f.readlines():
        if len(a)>=2:
            ip.append(a.strip().split(" "))
    
    cache={}
    for i in ip:
        cache[i[1]+'.']=i[0]

    #cache["abc.com."]="1.2.3.4"
    #cache["wrong.com."]="0.0.0.0"
    #cache["drdh.com."]="185.199.109.153"
    # 不良网站警告信息
    warning="illegal website or website doesn't exist"
   
    # ip需换成自己电脑的ip
    #server = Server((localIP, 53), Handler)
    server=socketserver.ThreadingUDPServer((localIP, 53),Handler)
    with server:
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        print('The DNS server is running at %s ...',localIP)
        server_thread.join()