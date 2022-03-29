# 此程式用來模擬實驗

# import Generator
from NSMQV.EmulatorParameters import EmulatorParameters
from NSMQV.Packet import Packet
from NSMQV.Generator import Generator
# from NSMQV.EmulatorParameters import queue_number
import numpy as np
from gym.utils import seeding


class NSMQV_Simulation(object):
    def __init__(self):
        self.queue_number = EmulatorParameters.queue_number
        self.theta = [0.01, 1, 0.05]  # 約束時間
        self.service_number = 3  # 服務類型數量
        # nsmqv_lambda[]   = [21, 6, 3] # 封包到達率 30
        # nsmqv_lambda[] = [42, 12, 6] # 60
        self.nsmqv_lambda = [63, 18, 9]  # 90 預設值
        # nsmqv_lambda = [84, 24, 12]  # 120
        # nsmqv_lambda = [105, 30, 15]  # 150
        # nsmqv_lambda = [126, 36, 18]  # 180
        # nsmqv_lambda = [147, 42, 21]  # 210
        self.total_lambda = self.nsmqv_lambda[0] + self.nsmqv_lambda[1] + self.nsmqv_lambda[2]
        self.r = [200, 500000, 75000]  # urllc, embb, mmtc packet UE2Edge input size (bits)
        self.w = [11000, 1000, 500]  # urllc, embb, mmtc packet Edge work load (cycles)
        self.o = [200, 500000, 75000]  # urllc, embb, mmtc packet Edge2UE output size ((bits)
        # 原資源數
        self.mu_UE = 40000000  # 總上傳頻寬 單位:bps
        self.mu_E = 4000000  # 總CPU 單位：KHz
        self.mu_EU = 63000000  # 總下載頻寬 單位: bps
        # / *
        # ---------------實驗頻寬資源漸增 -----------------
        # 總頻寬：50M 上：20M 下：30M
        # mu_UE = 20000000
        # mu_E = 4000000
        # mu_EU = 30000000
        # 總頻寬：75M 上：30M 下：45M
        # mu_UE = 30000000
        # mu_E = 4000000
        # mu_EU = 45000000
        # 總頻寬：100M 上：40M 下：60M
        # mu_UE = 40000000
        # mu_E = 4000000
        # mu_EU = 60000000
        # 總頻寬：125M 上：50M 下：75M
        # mu_UE = 50000000
        # mu_E = 4000000
        # mu_EU = 75000000
        # 總頻寬：150M 上：60M 下：90M
        # mu_UE = 60000000
        # mu_E = 4000000
        # mu_EU = 90000000
        # 總頻寬：175M 上：70M下：105M
        # mu_UE = 70000000
        # mu_E = 4000000
        # mu_EU = 105000000
        # 總頻寬：200M 上：80M 下：120M
        # mu_UE = 80000000
        # mu_E = 4000000
        # mu_EU = 120000000
        # 總頻寬：225M 上：90M 下：135M
        # mu_UE = 90000000
        # mu_E = 4000000
        # mu_EU = 135000000
        # 總頻寬：250M 上：100M 下：150M
        # mu_UE = 100000000
        # mu_E = 4000000
        # mu_EU = 150000000
        # * /
        # 各封包到達率之NS - MQV資源分配數 單位為: 原單位數(不是封包數) ------------------------------------------------------------------------
        # 封包到達率 = 調整
        self.mu_UE_j = [1.0849649711956881E7, 1.5998360758527812E7, 2.1151989529515307E7]  # 上傳頻寬的服務率(傳輸率)
        self.mu_E_j = [4693948.052696463, 29122.49841629079, 76929.44888724666]  # 在Edge的服務率(傳輸率)
        self.mu_EU_j = [2.44036944403482E7, 2.0173945508827038E7, 3.842236005082476E7]  # 下載頻寬的服務率(傳輸率)
        # self.mu_UE_j = [1.0849649711956881E7, 1.5998360758527812E7, 2.1151989529515307E7]  # 上傳頻寬的服務率(傳輸率)
        # self.mu_E_j = [4693948.052696463, 29122.49841629079, 76929.44888724666]  # 在Edge的服務率(傳輸率)
        # self.mu_EU_j = [2.44036944403482E7, 2.0173945508827038E7, 3.842236005082476E7]  # 下載頻寬的服務率(傳輸率)
        # 封包到達率 = 30
        # mu_UE_j = [1.0134302035019493E7, 9844835.931658013, 2.0020862033322494E7]  # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3911656.6953830062, 16927.695886698923, 71415.60873029483] # 在Edge的服務率(傳輸率)
        # mu_EU_j = [1.88328142517999E7, 1.26784423016607E7, 3.1488743446539402E7]  # 下載頻寬的服務率(傳輸率)
        # 封包到達率 = 60
        # mu_UE_j = [8810126.727228217, 1.2467412527895767E7, 1.8722460744876016E7]  # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3906064.751109772, 22798.588719751737, 71136.66017047642]  # 在Edge的服務率(傳輸率)
        # mu_EU_j = [1.727640690457293E7, 1.5405135451315546E7, 3.0318457644111525E7]  # 下載頻寬的服務率(傳輸率)
        # 封包到達率 = 90(預設值)
        # mu_UE_j = [7526969.140497902, 1.5070956049824556E7, 1.740207480967754E7]  # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3900370.3129043477, 28690.42662206324, 70939.26047358895]  # 在Edge的服務率(傳輸率)
        # mu_EU_j = [1.5710630600708619E7, 1.813612965305871E7, 2.915323974623267E7]  # 下載頻寬的服務率(傳輸率)
        # 封包到達率 = 120
        # mu_UE_j = [6292705.297550205, 1.7653521793847863E7, 1.6053772908601936E7]  # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3894534.261805938, 34614.28133119775, 70851.45686286408]  # 在Edge的服務率(傳輸率)
        # mu_EU_j = [1.4127584672668587E7, 2.0873985945716724E7, 2.799842938161469E7]  # 下載頻寬的服務率(傳輸率)
        # 封包到達率 = 150
        # mu_UE_j = [5124070.7081852835, 2.0210240792954423E7, 1.4665688498860294E7]  # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3888504.138135311, 40583.12341308542, 70912.73845160333]  # 在Edge的服務率(傳輸率)
        # mu_EU_j = [1.253179093083394E7, 2.361981323835319E7, 2.6848395830812868E7]  # 下載頻寬的服務率(傳輸率)

        # ---------------------------------------------------------------------------------------------
        # 均分方法分配數
        # mu_UE_j = [mu_UE / 3, mu_UE / 3, mu_UE / 3]  # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [mu_E / 3, mu_E / 3, mu_E / 3]  # 在Edge的服務率(傳輸率)
        # mu_EU_j = [mu_EU / 3, mu_EU / 3, mu_EU / 3]  # 下載頻寬的服務率(傳輸率)
        # / *
        # 實驗頻寬資源漸增之NS - MQV資源分配數 - ----------------
        # 總頻寬：50M 上：20M 下：30M
        # mu_UE_j = [5121631.544543765, 0.1276676125521366, 1.4878368327788621E7]  # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3924781.843973585, 3.48780584517917E-4, 75218.15567763429]  # 在Edge的服務率(傳輸率)
        # mu_EU_j = [8853154.423997581, 0.6175354663029229, 2.1146844958466955E7]  # 下載頻寬的服務率(傳輸率)
        # 總頻寬：75M 上：30M 下：45M
        # mu_UE_j = [3603579.617536649, 1.3793244610640299E7, 1.2603175771823052E7] # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3892224.2689320566, 30125.40041978252, 77650.33064816086] # 在Edge的服務率(傳輸率)
        # mu_EU_j = [7879095.165958313, 1.6279315145318205E7, 2.0841589688723482E7] # 下載頻寬的服務率(傳輸率)
        # 總頻寬：100M 上：40M 下：60M
        # mu_UE_j = [7487142.395985171, 1.508593716634122E7, 1.742692043767361E7] # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3900074.989779747, 28740.81984510879, 71184.19037514413] # 在Edge的服務率(傳輸率)
        # mu_EU_j = [1.456861926723852E7, 1.7769628815630317E7, 2.7661751917131163E7] # 下載頻寬的服務率(傳輸率)
        # 總頻寬：125M 上：50M 下：75M
        # mu_UE_j = [1.1807289248950643E7, 1.6135176776722223E7, 2.2057533974327132E7] # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3903444.204057036, 28204.97327820356, 68350.82266476048] # 在Edge的服務率(傳輸率)
        # mu_EU_j = [2.1261609797350135E7, 1.9157760766687956E7, 3.45806294359619E7] # 下載頻寬的服務率(傳輸率)
        # 總頻寬：150M 上：60M 下：90M
        # mu_UE_j = [1.6151686664983293E7, 1.7098348694187842E7, 2.6749964640828863E7] # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3905339.452091596, 27902.87311070721, 66757.67479769653] # 在Edge的服務率(傳輸率)
        # mu_EU_j = [2.7783492224052887E7, 2.0588006088849165E7, 4.1628501687097944E7] # 下載頻寬的服務率(傳輸率)
        # #
        # 封包到達率 = 175
        # mu_UE_j = [2.0453724029422425E7, 1.805991222914224E7, 3.1486363741435334E7] # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3906549.321289914, 27700.669893778853, 65750.00881630694] # 在Edge的服務率(傳輸率)
        # mu_EU_j = [3.420085237859928E7, 2.2062516014490996E7, 4.873663160690973E7] # 下載頻寬的服務率(傳輸率)
        # 封包到達率 = 200
        # mu_UE_j = [2.471106287332222E7, 1.9042871996945333E7, 3.6246065129732445E7] # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3907391.0533113996, 27554.21740034144, 65054.729288259055] # 在Edge的服務率(傳輸率)
        # mu_EU_j = [4.055972433795849E7, 2.3578396780364398E7, 5.586187888167711E7] # 下載頻寬的服務率(傳輸率)
        # 封包到達率 = 225
        # mu_UE_j = [2.89361171816274E7, 2.0043491541103557E7, 4.102039127726905E7] # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3908010.0746921184, 27443.679079878126, 64546.24622800341] # 在Edge的服務率(傳輸率)
        # mu_EU_j = [4.688191015863299E7, 2.5118775197297875E7, 6.2999314644069135E7] # 下載頻寬的服務率(傳輸率)
        # 封包到達率 = 250
        # mu_UE_j = [3.314669442572432E7, 2.10653190403727E7, 4.578798653390298E7] # 上傳頻寬的服務率(傳輸率)
        # mu_E_j = [3908480.8326213984, 27356.557763434474, 64162.609615167145] # 在Edge的服務率(傳輸率)
        # mu_EU_j = [5.315929173909351E7, 2.668539304434387E7, 7.015531521656261E7] # 下載頻寬的服務率(傳輸率)
        # * /

        self.number_packet = 100000         # 模擬封包數
        self.packet = []  # 建封包
        # for i in range(self.number_packet):
        #     self.packet[i] = Packet()
        # self.packet = Packet[self.number_packet]  # 建封包

        self.qos_violation = 0.0

        self.step_change = [self.mu_UE * 0.01, self.mu_E * 0.01, self.mu_EU * 0.01]

    def resource_allocate(self, mu_UE, mu_E, mu_EU):
        self.mu_UE_j = mu_UE  # 設定上傳頻寬的服務率(傳輸率)
        self.mu_E_j = mu_E  # 設定在Edge的服務率(傳輸率)
        self.mu_EU_j = mu_EU  # 設定下載頻寬的服務率(傳輸率)

    def initial_packet(self):
        self.packet = []  # reset packet
        # Random number generator
        g = Generator()

        # 各服務當下時間點 embb mmtc不是設為0是因為會先服務最先到的urllc之後去比較embb跟mmtc的第一個封包時間去決定是否要換生產不同服務類型
        current_time = [0, g.getRandomNumber_Exponential(1 / self.nsmqv_lambda[1]),
                        g.getRandomNumber_Exponential(1 / self.nsmqv_lambda[2])]
        service_type_gen = 0
        # 產生設定封包數個封包
        for i in range(self.number_packet):
            inter_arrival_time = g.getRandomNumber_Exponential(
                1 / self.nsmqv_lambda[service_type_gen])  # 每個封包多久會來一次=卜瓦松過程(=指數分布的倒數)封包率
            current_time[service_type_gen] += inter_arrival_time  # 累加封包到達時間(為某一服務到達時間累加的時間)
            arrival_time = current_time[service_type_gen]  # 此封包到達時間設為累加過後的現在時間
            current_time_gen = current_time[service_type_gen]  # 用此參數記住目前服務的現在時間以去跟別的服務比較

            # 目前服務類型之現在時間累計 > 其他服務現在時間累計 = > 換產生別種服務類型封包
            for j in range(self.service_number):
                if current_time_gen > current_time[j]:
                    service_type_gen = j  # 更換服務類型
                    current_time_gen = current_time[service_type_gen]  # 目前時間改成更換之服務類型的時間累計
                    arrival_time = current_time[service_type_gen]  # 到達時間改成該服務累計時間

            r_task_size = self.r[service_type_gen]  # 設定此封包任務輸入大小為此服務的任務輸入大小
            w_task_size = self.w[service_type_gen]  # 設定此封包任務工作負載大小為此服務的工作負載大小
            o_task_size = self.o[service_type_gen]  # 設定此封包任務輸出大小為此服務的任務輸出大小

            service_time = [0]*self.queue_number  # 初始服務時間為一個陣列

            service_time[0] = g.getRandomNumber_Exponential(r_task_size) / self.mu_UE_j[
                service_type_gen]  # 任務輸入時間=任務輸入大小 / 該服務分配到的上傳頻寬
            service_time[1] = g.getRandomNumber_Exponential(w_task_size) / self.mu_E_j[
                service_type_gen]  # 任務工作負載時間=任務工作負載大小 / 該服務分配到的CPU
            service_time[2] = g.getRandomNumber_Exponential(o_task_size) / self.mu_EU_j[
                service_type_gen]  # 任務輸出時間=任務輸出大小 / 該服務分配到的下載頻寬
            # 只切 E
            # service_time[0] = g.getRandomNumber_Exponential(r_task_size) / mu_UE
            # service_time[1] = g.getRandomNumber_Exponential(w_task_size) / mu_E_j[service_type_gen]
            # service_time[2] = g.getRandomNumber_Exponential(o_task_size) / mu_EU
            # # 只切 UE EU
            # service_time[0] = g.getRandomNumber_Exponential(r_task_size) / (mu_UE_j[service_type_gen])
            # service_time[1] = g.getRandomNumber_Exponential(w_task_size) / (mu_E)
            # service_time[2] = g.getRandomNumber_Exponential(o_task_size) / (mu_EU_j[service_type_gen])
            # 只切UE
            # service_time[0] = g.getRandomNumber_Exponential(r_task_size) / (mu_UE_j[service_type_gen])
            # service_time[1] = g.getRandomNumber_Exponential(w_task_size) / (mu_E)
            # service_time[2] = g.getRandomNumber_Exponential(o_task_size) / (mu_EU)
            # 只切EU
            # service_time[0] = g.getRandomNumber_Exponential(r_task_size) / (mu_UE)
            # service_time[1] = g.getRandomNumber_Exponential(w_task_size) / (mu_E)
            # service_time[2] = g.getRandomNumber_Exponential(o_task_size) / (mu_EU_j[service_type_gen])
            # # 只切UE E
            # service_time[0] = g.getRandomNumber_Exponential(r_task_size) / (mu_UE_j[service_type_gen])
            # service_time[1] = g.getRandomNumber_Exponential(w_task_size) / (mu_E_j[service_type_gen])
            # service_time[2] = g.getRandomNumber_Exponential(o_task_size) / (mu_EU)
            # # 只切 E EU
            # service_time[0] = g.getRandomNumber_Exponential(r_task_size) / (mu_UE)
            # service_time[1] = g.getRandomNumber_Exponential(w_task_size) / (mu_E_j[service_type_gen])
            # service_time[2] =  g.getRandomNumber_Exponential(o_task_size) / (mu_EU_j[service_type_gen])

            deadline_time = arrival_time + self.theta[service_type_gen]  # 此封包的期限時間設=到達時間+此封包服務類別的延遲限制

            self.packet.append(Packet(i + 1, service_type_gen, r_task_size, w_task_size, o_task_size,
                                                  arrival_time, service_time, deadline_time))  # 利用packet函數產生封包

    def m_m_1_NS(self, k):  # k為(資源)queue的編號0~2
        isNotLast = bool(k != self.queue_number - 1)  # 判斷是否處理到最後一queue
        previous_packet_departure_time = 0  # 前一封包離開時間初始為0

        # 將產生的封包依照arrival_time大小排序
        for i in range(self.number_packet):
            for j in range(i):
                if self.packet[j].arrival_time[k] > self.packet[i].arrival_time[k]:
                    tmp = self.packet[j]
                    self.packet[j] = self.packet[i]
                    self.packet[i] = tmp

        # 處理每一個封包
        for i in range(self.number_packet):  # i為封包編號
            self.packet[i].waiting_queue_length[k] = 0  # 初始封包目前所在佇列的等待排隊長度為=0

            # 需等候(到達時間 < 前一封包離開時間)
            if self.packet[i].arrival_time[k] < previous_packet_departure_time:
                self.packet[i].departure_time[k] = previous_packet_departure_time + self.packet[i].service_time[
                    k]  # 離開時間=上一個封包離開時間+服務時間
                self.packet[i].waiting_time[k] = previous_packet_departure_time - self.packet[i].arrival_time[
                    k]  # 等待時間=離開時間-到達時間

                # j為i(目前封包)前面所有的封包
                for j in range(i):
                    if self.packet[i].arrival_time[k] < self.packet[j].departure_time[k]:  # 如果到達時間比前面任一封包離開時間早
                        self.packet[i].waiting_queue_length[k] += 1  # 佇列等待排隊長度+1
                    else:
                        break
            # 不需等候 = > 封包一到就處理
            else:
                self.packet[i].departure_time[k] = self.packet[i].arrival_time[k] + self.packet[i].service_time[
                    k]  # 離開時間 = 到達時間 + 服務時間
                self.packet[i].waiting_time[k] = 0

            # 更新前一個封包離開時間為目前此封包離開時間
            previous_packet_departure_time = self.packet[i].departure_time[k]
            if isNotLast:  # 如封包還沒到最後一queue = > 下一queue進入時間 = 離開此queue的時間
                self.packet[i].arrival_time[k + 1] = self.packet[i].departure_time[k]

    def m_m_1_NSMQV(self, k):  # k為queue number 與NS不同之處主要是NS - MQV是三個服務類別個別處理，所以有些設值會分開去計算。ex：前一封包離開時間等等
        isNotLast = bool(k != self.queue_number - 1)
        previous_packet_departure_time = [0, 0, 0]  # NS - MQV是三個服務個別處理，所以設為陣列

        # 處理每一個封包
        for i in range(self.number_packet):
            self.packet[i].waiting_queue_length[k] = 0

            # 需等候 (到達時間 < 前一封包離開時間)
            if self.packet[i].arrival_time[k] < previous_packet_departure_time[self.packet[i].service_type]:
                self.packet[i].departure_time[k] = previous_packet_departure_time[self.packet[i].service_type] + \
                                                   self.packet[i].service_time[k]
                self.packet[i].waiting_time[k] = previous_packet_departure_time[self.packet[i].service_type] - \
                                                 self.packet[i].arrival_time[k]

                # j為i前面的封包
                for j in range(i):
                    if self.packet[i].arrival_time[k] < self.packet[j].departure_time[k]:
                        self.packet[i].waiting_queue_length[k] += 1
                    else:
                        break

            # 不需等候
            else:
                self.packet[i].departure_time[k] = self.packet[i].arrival_time[k] + self.packet[i].service_time[k]
                self.packet[i].waiting_time[k] = 0

            # 更新前一個封包離開時間
            previous_packet_departure_time[self.packet[i].service_type] = self.packet[i].departure_time[k]
            if isNotLast:
                self.packet[i].arrival_time[k + 1] = previous_packet_departure_time[self.packet[i].service_type]

    def summary(self):
        total_service_time = [0]*self.queue_number
        total_waiting_time = [0]*self.queue_number
        total_processing_time = 0
        simulation_time = self.packet[self.number_packet - 1].departure_time[self.queue_number - 1]
        QV = 0  # 計算已延遲封包(超過deadline_time)
        total_delay_packet_type0 = 0  # service type urllc 超過deadline
        total_delay_packet_type1 = 0  # service type embb 超過deadline
        total_delay_packet_type2 = 0  # service type mmtc 超過deadline
        total_packet_type0 = 0  # service type urllc
        total_packet_type1 = 0  # service type embb
        total_packet_type2 = 0  # service type mmtc
        over_deadline_time = 0
        total_service_time_UE = [0]*self.queue_number
        total_service_time_E = [0]*self.queue_number
        total_service_time_EU = [0]*self.queue_number
        # total_waiting_time_UE = [queue_number]
        # total_waiting_time_E = [queue_number]
        # total_waiting_time_EU = [queue_number]

        # 顯示和統計每一個封包的時間
        for i in range(self.number_packet):
            if self.packet[i].service_type == 0:
                total_packet_type0 += 1  # 加總urllc服務時間
            if self.packet[i].service_type == 1:
                total_packet_type1 += 1  # 加總embb服務時間
            if self.packet[i].service_type == 2:
                total_packet_type2 += 1  # 加總mmtc服務時間

            total_service_time_UE[self.packet[i].service_type] += self.packet[i].service_time[0]  # 0代表queue0= > UE
            total_service_time_E[self.packet[i].service_type] += self.packet[i].service_time[1]  # 1代表queue1= > E
            total_service_time_EU[self.packet[i].service_type] += self.packet[i].service_time[2]  # 2代表queue2= > EU
            # total_waiting_time_UE[packet[i].service_type] += packet[i].waiting_time[0]
            # total_waiting_time_E[packet[i].service_type] += packet[i].waiting_time[1]
            # total_waiting_time_EU[packet[i].service_type] += packet[i].waiting_time[2]

            for k in range(self.queue_number):
                total_service_time[k] += self.packet[i].service_time[k]  # 計算某一queue的服務時間
                total_waiting_time[k] += self.packet[i].waiting_time[k]  # 計算某一queue的等待時間

                # 違反機率只要看各封包最後一個queue的離開時間是否大於期限時間 大於= > 違反的封包數+1
                if k == 2:
                    if self.packet[i].departure_time[k] > self.packet[i].deadline_time:
                        QV += 1  # 總體違反的封包數
                if self.packet[i].service_type == 0:
                    total_delay_packet_type0 += 1  # urllc違反的封包數
                if self.packet[i].service_type == 1:
                    total_delay_packet_type1 += 1  # embb違反的封包數
                if self.packet[i].service_type == 2:
                    total_delay_packet_type2 += 1  # mmtc違反的封包數

            total_processing_time += self.packet[i].departure_time[self.queue_number - 1] - self.packet[i].arrival_time[
                0]  # 計算總體處理時間

            # print('封包(' + str(i + 1) + '): ')  # 顯示封包run的資訊
            # print(self.packet[i])

        # 顯示統計結果
        # for k in range(queue_number):
        # print('Queue_' + (k + 1) + '服務時間: ' + (total_service_time[k] / number_packet))
        # print('Queue_' + (k + 1) + '等候時間: ' + (total_waiting_time[k] / number_packet))
        # print('Queue_' + (k + 1) + '等候長度: ' + (total_waiting_time[k] / simulation_time))

        # print('urllc超過deadline的封包數: ', total_delay_packet_type0)
        # print('urllc總封包數: ' + str(total_packet_type0))
        # print('urllc封包超過deadline的比例: ' + str(total_delay_packet_type0 / total_packet_type0))
        # print('embb超過deadline的封包數: ' + str(total_delay_packet_type1))
        # print('embb總封包數: ' + str(total_packet_type1))
        # print('embb封包超過deadline的比例: ' + str(total_delay_packet_type1 / total_packet_type1))
        # print('mmtc超過deadline的封包數: ' + str(total_delay_packet_type2))
        # print('mmtc總封包數: ' + str(total_packet_type2))
        # print('mmtc封包超過deadline的比例: ' + str(total_delay_packet_type2 / total_packet_type2))
        # print('超過deadline的比例QV: ' + str(QV / self.number_packet))  # 違反機率=總體違反封包數 / 總體封包數

        self.qos_violation = QV / self.number_packet

    def output_QV(self, alloc_change):
        self.resource_allocate(alloc_change[:3], alloc_change[3:6], alloc_change[6:])

        # 初始化封包
        self.initial_packet()

        # 改切割方法 - -------------------------------------------
        for k in range(self.queue_number):  # k為(資源)queue編號 0~2 0:UE 1:E 2:EU
            # m_m_1_NS(k)  # 用NS(沒切割FIFO)方法切割
            self.m_m_1_NSMQV(k)  # 用NS - MQV切割

        self.summary()  # 顯示統計結果(在程式最下面)可改動想顯示的資料

        return self.qos_violation

    def step(self, action_type):
        # # 回合(Episode)結束
        # if self.qos_violation < 0.01:
        #     raise Exception("Game is over")

        state = []
        mu_all = np.array([self.mu_UE_j, self.mu_E_j, self.mu_EU_j], dtype=float)

        for i in range(3):

            # for i in range(len(mu_all[mu_i][:])):
            #     if self.step_change > mu_all[mu_i][i]:
            #         self.step_change = mu_all[mu_i][i] / 2
            #     elif self.step_change < 10:
            #         self.step_change = self.step_change * 2

            # if self.step_change < 10:
            #     self.step_change = self.step_change * 2
            # else:
            #     self.step_change = self.step_change / 2

            state = np.append(state, self.action_types(action_type[i], self.step_change[i] * self.qos_violation))

        done = bool(
            self.qos_violation < 0.015
        )

        # 減少1步
        # self.steps_left -= 1

        """
        Actions:
            Type: Discrete(6)
            Num   Action
            0     [+u, -u, 0]
            1     [+u, 0, -u]
            2     [0, +u, -u]
            3     [0, -u, +u]
            4     [-u, +u, 0]
            5     [-u, 0, +u]
        """

        # self.state = np.hstack((action_types[action_type[0]][:], action_types[action_type[1]][:], action_types[action_type[2]][:]))

        # 隨機策略，任意行動，並給予獎勵(亂數值)
        return state, self.qos_violation, done

    def action_types(self, action_type, step_change):

        # action_types = np.array([[self.step_change, -self.step_change, 0],
        #                          [self.step_change, 0, -self.step_change],
        #                          [0, self.step_change, -self.step_change],
        #                          [0, -self.step_change, self.step_change],
        #                          [-self.step_change, self.step_change, 0],
        #                          [-self.step_change, 0, self.step_change]], dtype=float)
        # 分配量用排列組合方式選擇
        # action_types = np.array([[step_change, -step_change, 0],
        #                          [step_change, 0, -step_change],
        #                          [0, step_change, -step_change],
        #                          [0, -step_change, step_change],
        #                          [-step_change, step_change, 0],
        #                          [-step_change, 0, step_change]], dtype=float)

        """
        Actions:
                Num   Action
                0     [+u, 0, -u]
                1     [+u, -u, 0]
                2     [0, +u, -u]
                3     [-u, +u, 0]
                4     [0, -u, +u]
                5     [-u, 0, +u]
        """

        action_types = np.array([[step_change, 0, -step_change],
                                 [step_change, -step_change, 0],
                                 [0, step_change, -step_change],
                                 [-step_change, step_change, 0],
                                 [0, -step_change, step_change],
                                 [-step_change, 0, step_change]], dtype=float)

        return action_types[action_type][:]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

if __name__ == '__main__':
    env = NSMQV_Simulation()

    # alloc_UE = [1.0849649711956881E7, 1.5998360758527812E7, 2.1151989529515307E7]  # 上傳頻寬的服務率(傳輸率)
    # alloc_E = [4693948.052696463, 29122.49841629079, 76929.44888724666]  # 在Edge的服務率(傳輸率)
    # alloc_EU = [2.44036944403482E7, 2.0173945508827038E7, 3.842236005082476E7]  # 下載頻寬的服務率(傳輸率)
    alloc_UE = [40000000/3, 40000000/3, 240000000/3]  # 上傳頻寬的服務率(傳輸率)
    alloc_E = [4000000/3, 4000000/3, 4000000/3]  # 在Edge的服務率(傳輸率)
    alloc_EU = [63000000/3, 63000000/3, 63000000/3]  # 下載頻寬的服務率(傳輸率)
    env.resource_allocate(alloc_UE, alloc_E, alloc_EU)

    # 初始化封包
    env.initial_packet()

    # 改切割方法 - -------------------------------------------
    for k in range(env.queue_number):  # k為(資源)queue編號 0~2 0:UE 1:E 2:EU
        # m_m_1_NS(k)  # 用NS(沒切割FIFO)方法切割
        env.m_m_1_NSMQV(k)  # 用NS - MQV切割
#     # 只切E
#     # m_m_1_NS(0)
#     # m_m_1_NSMQV(1)
#     # m_m_1_NS(2)
#     # 只切UE EU
#     # m_m_1_NSMQV(0)
#     # m_m_1_NS(1)
#     # m_m_1_NSMQV(2)
#     # 只切UE
#     # m_m_1_NSMQV(0)
#     # m_m_1_NS(1)
#     # m_m_1_NS(2)
#     # 只切EU
#     # m_m_1_NS(0)
#     # m_m_1_NS(1)
#     # m_m_1_NSMQV(2)
#     # 只切UE E
#     # m_m_1_NSMQV(0)
#     # m_m_1_NSMQV(1)
#     # m_m_1_NS(2)
#     # 只切E EU
#     # m_m_1_NS(0)
#     # m_m_1_NSMQV(1)
#     # m_m_1_NSMQV(2)
#     # --------------------------------------------------
#
    env.summary()  # 顯示統計結果(在程式最下面)可改動想顯示的資料
# #
# #     print('*************************************\n')
# #     print(QoS_violation + '\n')
# #     #
# #     # try [
# #     # Thread.sleep(1000)
# #     #	] catch (InterruptedException e) [
# #     #			# TODO Auto-generated catch block
# #     #			e.printStackTrace()
# #     #		]
# #     #		Output_QV()
# #     # ]
# #
# # QoS_violation = 0
# # def Output_QV:
# #     # 初始化封包
# #     initial_packet()
# #
# #     # 改切割方法 --------------------------------------------
# #     for k in range(NSMQV_Simulation.queue_number):  # k為(資源)queue編號 0~2 0:UE 1:E 2:EU
# #         # m_m_1_NS(k)  # 用NS(沒切割FIFO)方法切割
# #         m_m_1_NSMQV(k)  # 用NS - MQV切割
# #
# #     summary() # 顯示統計結果(在程式最下面)可改動想顯示的資料
# #
# # print('*************************************\n')
# # print(QoS_violation + '\n')
# #
# # return QoS_violation