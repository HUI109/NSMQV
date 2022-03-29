from NSMQV.Packet import Packet

num = 2
packet = []
for i in range(num):
    # packet[i] = Packet
    p = Packet(i+1, 2, 200, 100, 300, [10, 20, 30], [0, 1, 2], 5)
    packet.append(p)
    print(packet[i])

# print(packet[0])
# print(packet[1])
