# 創建封包

from NSMQV.EmulatorParameters import EmulatorParameters


class Packet(object):
    def __init__(self, packet_index, service_type, r_task_size, w_task_size, o_task_size, arrival_time_gen, service_time_gen, deadline_time_gen):
        self.queue_number = EmulatorParameters.queue_number
        self.service_number = EmulatorParameters.service_number
        self.arrival_time = [0]*self.queue_number  # arrival_time
        self.service_time = [0]*self.queue_number  # service_time
        self.departure_time = [0]*self.queue_number  # departure_time
        self.waiting_time = [0]*self.queue_number  # waiting_time
        self.waiting_queue_length = [0]*self.queue_number  # waiting_queue_length
        self.waiting_queue_length_scheduled = [0]*self.queue_number  # waiting_queue_length after scheduling
        self.packet_index = 0  # packet_index
        self.service_type = 0  # service_type
        self.r_task_size = 0  # 任務大小(UE傳輸)
        self.w_task_size = 0  # 封包大小(Edge運算)
        self.o_task_size = 0  # 封包大小(EU傳輸)
        self.deadline_time = 0  # deadline_time
        self.constant_deadline_time = [0]*self.service_number  # deadline_time

    # def packet(self, packet_index, service_type, r_task_size, w_task_size, o_task_size, arrival_time_gen, service_time_gen, deadline_time_gen):

        self.packet_index = packet_index
        self.service_type = service_type
        # for k in range(self.queue_number):
        #     self.arrival_time.append(0)
        self.service_time = service_time_gen
        #     self.departure_time.append(0)
        #     self.waiting_time.append(0)
        #     self.waiting_queue_length.append(0)
        #     self.waiting_queue_length_scheduled.append(0)
        #     self.constant_deadline_time.append(0)

        self.constant_deadline_time[service_type] = 0
        self.arrival_time[0] = arrival_time_gen
        self.deadline_time = deadline_time_gen
        self.r_task_size = r_task_size
        self.w_task_size = w_task_size
        self.o_task_size = o_task_size

    def __str__(self):
        buffer = ''
        for i in range(self.queue_number):
            buffer += '---Queue_' + str(i + 1) + '---\n'
            buffer += '封包編號: ' + str(self.packet_index) + '\n'
            buffer += '服務類型: ' + str(self.service_type) + '\n'
            buffer += 'r, w, o: ' + str(self.r_task_size) + ', ' + str(self.w_task_size) + ', ' + str(self.o_task_size) + '\n'
            buffer += '到達時間: ' + str(self.arrival_time[i]) + '\n'
            buffer += '服務時間: ' + str(self.service_time[i]) + '\n'
            buffer += '完成時間: ' + str(self.departure_time[i]) + '\n'
            buffer += '截止時間: ' + str(self.deadline_time) + '\n'
            buffer += '等候時間: ' + str(self.waiting_time[i]) + '\n'
            buffer += '等候長度: ' + str(self.waiting_queue_length[i]) + '\n'
            buffer += '排程後等候長度: ' + str(self.waiting_queue_length_scheduled[i]) + '\n'

        return buffer

