from NSMQV.NSMQV_Simulation import NSMQV_Simulation
import random
import gym
import torch


class Environment:
    def __init__(self):
        self.steps_left = 1

    def get_observation(self):
        # 狀態空間(State Space)
        return [0.0, 0.5, 1.0]

    def get_actions(self):
        # 行動空間(Action Space)
        return [0, 1]

    def is_done(self):
        # 回合(Episode)是否結束
        return self.steps_left <= 0.01

    # 步驟
    def step(self, action):
        # 回合(Episode)結束
        if self.is_done():
            raise Exception("Game is over")

        # 減少1步
        self.steps_left -= 1

        # 隨機策略，任意行動，並給予獎勵(亂數值)
        return random.choice(self.get_observation()), random.random()


class Agent:
    # 初始化
    def __init__(self):
        pass

    def action(self, env):
        # 觀察或是取得狀態
        current_obs = env.get_observation()
        # 採取行動
        actions = env.get_actions()
        return random.choice(actions)


if __name__ == "__main__":
    # region
    # 實驗
    envNS = NSMQV_Simulation()

    # 建立環境、代理人物件
    envRL = Environment()
    agent = Agent()

    # # 累計報酬
    # total_reward = 0
    # while not envRL.is_done():
    #     # 採取行動
    #     action = agent.action(envRL)
    #
    #     # 進到下一步
    #     state, reward = envRL.step(action)
    #
    #     # 報酬累計
    #     # print(reward)
    #     total_reward += reward
    #
    # # 顯示累計報酬
    # print(f"累計報酬: {total_reward:.4f}")

    mu_UE_j = [1.0849649711956881E7, 1.5998360758527812E7, 2.1151989529515307E7]  # 上傳頻寬的服務率(傳輸率)
    mu_E_j = [4693948.052696463, 29122.49841629079, 76929.44888724666]  # 在Edge的服務率(傳輸率)
    mu_EU_j = [2.44036944403482E7, 2.0173945508827038E7, 3.842236005082476E7]  # 下載頻寬的服務率(傳輸率)
    envNS.resource_allocate(mu_UE_j, mu_E_j, mu_EU_j)

    # 初始化封包
    envNS.initial_packet()

    # 改切割方法 - -------------------------------------------
    for k in range(envNS.queue_number):  # k為(資源)queue編號 0~2 0:UE 1:E 2:EU
        # m_m_1_NS(k)  # 用NS(沒切割FIFO)方法切割
        envNS.m_m_1_NSMQV(k)  # 用NS - MQV切割

    envNS.summary()  # 顯示統計結果(在程式最下面)可改動想顯示的資料
    # endregion
