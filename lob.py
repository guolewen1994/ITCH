import os
import glob
import csv
from collections import defaultdict
import itertools
import pandas as pd
import shutil


class ConstructLOB:
    def __init__(self):
        self._lob = []
        self._order_aggressiveness = []
        self._bid_info = defaultdict(list)
        self._bid_price = []
        self._bid_order_info = {}
        self._ask_info = defaultdict(list)
        self._ask_price = []
        self._ask_order_info = {}

    def reset(self):
        self.__init__()

    def add_2_lob(self, stock, timestamp, order_ref, buy_sell, shares, price, mpid="NSDQ", level=10):
        order_aggressiveness = None
        if buy_sell == "B":
            if not self._bid_price or price > self._bid_price[0]:
                order_aggressiveness = 4
            elif price == self._bid_price[0]:
                order_aggressiveness = 5
            elif price < self._bid_price[0]:
                order_aggressiveness = 6
            self._bid_order_info[order_ref] = [price, shares, mpid, order_aggressiveness]            
            self._bid_price.append(price)
            self._bid_price = sorted(set(self._bid_price), reverse=True)
            self._bid_info[price].append(order_ref)

        elif buy_sell == "S":
            if not self._ask_price or price < self._ask_price[0]:
                order_aggressiveness = 4
            elif price == self._ask_price[0]:
                order_aggressiveness = 5
            elif price > self._ask_price[0]:
                order_aggressiveness = 6
            self._ask_order_info[order_ref] = [price, shares, mpid, order_aggressiveness]
            self._ask_price.append(price)
            self._ask_price = sorted(set(self._ask_price))
            self._ask_info[price].append(order_ref)
        self.update_aggressiveness(stock, timestamp, order_ref, order_aggressiveness)
        self.update_lob(stock, timestamp, level=level)
        return None

    def delete_2_lob(self, stock, timestamp, order_ref, level=5):
        order_aggressiveness = None
        try:
            if self._bid_order_info[order_ref][0] == self._bid_price[0]:
                order_aggressiveness = 9
            elif self._bid_order_info[order_ref][0] < self._bid_price[0]:
                order_aggressiveness = 10
            price, shares, ticker, order_aggressiveness = self._bid_order_info.pop(order_ref)
            self._bid_info[price].remove(order_ref)
            if not self._bid_info[price]:
                self._bid_info.pop(price)
                self._bid_price = sorted(list(self._bid_info.keys()), reverse=True)

        except KeyError:
            price, shares, ticker, order_aggressiveness = self._ask_order_info.pop(order_ref)
            self._ask_info[price].remove(order_ref)
            if not self._ask_info[price]:
                self._ask_info.pop(price)
                self._ask_price = sorted(list(self._ask_info.keys()))
        self.update_lob(stock, timestamp, level=level)
        return

    def revise_2_lob(self, stock, timestamp, ori_ref_num, new_ref_num, shares, price, level=5):
        try:
            self._bid_order_info[new_ref_num] = self._bid_order_info.pop(ori_ref_num)
            ori_price = self._bid_order_info[new_ref_num][0]
            self._bid_info[ori_price].remove(ori_ref_num)
            # check if the price level exists or not
            if not self._bid_info[ori_price]:
                self._bid_info.pop(ori_price)
                self._bid_price = sorted(list(self._bid_info.keys()), reverse=True)

            self._bid_order_info[new_ref_num][0] = price
            self._bid_order_info[new_ref_num][1] = shares
            self._bid_price.append(price)
            self._bid_price = sorted(set(self._bid_price), reverse=True)
            self._bid_info[price].append(new_ref_num)
            
        except KeyError:
            self._ask_order_info[new_ref_num] = self._ask_order_info.pop(ori_ref_num)
            # remove the order at original price
            ori_price = self._ask_order_info[new_ref_num][0]
            self._ask_info[ori_price].remove(ori_ref_num)
            # check if the price level exists or not
            if not self._ask_info[ori_price]:
                self._ask_info.pop(ori_price)
                self._ask_price = sorted(list(self._ask_info.keys()))

            self._ask_order_info[new_ref_num][0] = price
            self._ask_order_info[new_ref_num][1] = shares
            self._ask_price.append(price)
            self._ask_price = sorted(set(self._ask_price))
            self._ask_info[price].append(new_ref_num)
        self.update_lob(stock, timestamp, level=level)
        return

    def cancel_2_lob(self, stock, timestamp, order_ref, cancel_shares, level=5):
        try:
            self._bid_order_info[order_ref][1] = self._bid_order_info[order_ref][1] - cancel_shares
        except KeyError:
            self._ask_order_info[order_ref][1] = self._ask_order_info[order_ref][1] - cancel_shares
        self.update_lob(stock, timestamp, level=level)
        return

    def execution_2_lob(self, stock, timestamp, order_ref, trade_size, level=5):
        try:
            self._bid_order_info[order_ref][1] = self._bid_order_info[order_ref][1] - trade_size
            # if shares for this order become 0, then delete the order
            if self._bid_order_info[order_ref][1] == 0:
                order_info = self._bid_order_info.pop(order_ref)
                price, mpid = order_info[0], order_info[2]
                self._bid_info[order_info[0]].remove(order_ref)
                # search the price to order infor, if no order at this price level delete this price level
                if not self._bid_info[price]:
                    self._bid_info.pop(price)
                    self._bid_price = sorted(list(self._bid_info.keys()), reverse=True)
        except KeyError:
            self._ask_order_info[order_ref][1] = self._ask_order_info[order_ref][1] - trade_size
            # if shares for this order become 0, then delete the order
            if self._ask_order_info[order_ref][1] == 0:
                order_info = self._ask_order_info.pop(order_ref)
                price, mpid = order_info[0], order_info[2]
                self._ask_info[order_info[0]].remove(order_ref)
                # search the price to order infor, if no order at this price level delete this price level
                if not self._ask_info[price]:
                    self._ask_info.pop(price)
                    self._ask_price = sorted(list(self._ask_info.keys()))
        self.update_lob(stock, timestamp, level=level)

    def update_lob(self, stock, timestamp, level=10):
        l = [None] * (level * 6 + 2)
        l[0] = stock
        l[1] = timestamp
        # if the current prices available larger than the level, only take the first best level quotes
        # else take all the quotes
        temp_bid_price_l = []
        temp_ask_price_l = []
        if len(self._bid_price) == 0:
            pass
        elif len(self._bid_price) < level:
            temp_bid_price_l = self._bid_price
        elif len(self._bid_price) >= level:
            # if the length of the bid prices exceeds the level we need we slice the list to size "level"
            temp_bid_price_l = list(itertools.islice(self._bid_price, level))
            
        if len(self._bid_price) == 0:
            pass
        else:
            for i, v in enumerate(temp_bid_price_l):
                l[3 * i + 2] = v
                # get the orders at this price v then search the order in the order_info
                # sum the order size as the depth
                l[3 * i + 3] = sum([self._bid_order_info[order][1] for order in self._bid_info[v]])
                l[3 * i + 4] = ";".join([self._bid_order_info[order][2] + "*" 
                                         + str(self._bid_order_info[order][1]) for order in self._bid_info[v]])
        if len(self._ask_price) == 0:
            pass
        elif len(self._ask_price) < level:
            temp_ask_price_l = self._ask_price
        elif len(self._ask_price) >= level:
            # if the length of the ask prices exceeds the level we need we slice the list to size "level"
            temp_ask_price_l = list(itertools.islice(self._ask_price, level))
        
        # iterating the temp_price_l to get the information for each price level
        if len(self._ask_price) == 0:
            pass
        else:
            for i, v in enumerate(temp_ask_price_l):
                l[3 * i + level * 3 + 2] = v
                # get the orders at this price v then search the order in the order_info
                # sum the order size as the depth
                l[3 * i + level * 3 + 3] = sum([self._ask_order_info[order][1] for order in self._ask_info[v]])
                l[3 * i + level * 3 + 4] = ";".join([self._ask_order_info[order][2] + "*" 
                                                + str(self._ask_order_info[order][1]) for order in self._ask_info[v]])
                # join the mpid for the orders l[3 * i + 3]!!!!
        self._lob.append(l)
        return

    def update_aggressiveness(self, stock, timestamp, order_ref, order_aggressiveness):
        self._order_aggressiveness.append([stock, timestamp, order_ref, order_aggressiveness])

    def construct(self, in_path, out_path, level=10):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        os.chdir(out_path)
        stock = 1
        path = sorted(glob.glob(in_path + "/*.csv"))
        for file in path:
            with open(file, "r") as f:
                files = csv.reader(f, delimiter=',')
                for line in files:
                    # check the stock is the same or not
                    if stock != int(line[1]):
                        print("finish stock {}".format(stock))
                        pd.DataFrame(self._lob).to_csv(os.path.join(out_path, "{}.csv".format(stock)), index=False)
                        pd.DataFrame(self._order_aggressiveness).to_csv(os.path.join(out_path, "aggr_{}.csv".format(stock)), index=False)
                        stock = int(line[1])
                        self.reset()
                    else:
                        pass

                    if line[0] == "A":
                        self.add_2_lob(int(line[1]), int(float(line[3])), line[4], line[5],
                                       int(line[6]), float(line[8])/10000, level=level)
                    elif line[0] == "F":
                        self.add_2_lob(int(line[1]), int(float(line[3])), line[4], line[5],
                                       int(line[6]), float(line[8])/10000, mpid=line[9], level=level)
                    elif line[0] == "D":
                        self.delete_2_lob(int(line[1]), int(float(line[3])), line[4], level=level)
                    elif line[0] == "U":
                        self.revise_2_lob(int(line[1]), int(float(line[3])), line[15],
                                          line[16], int(line[6]), float(line[8]) / 10000, level=level)
                    elif line[0] == "X":
                        self.cancel_2_lob(int(line[1]), int(float(line[3])), line[4], int(line[14]), level=level)
                    elif line[0] == "E":
                        self.execution_2_lob(int(line[1]), int(float(line[3])), line[4], int(line[10]), level=level)
                    elif line[0] == "C":
                        self.execution_2_lob(int(line[1]), int(float(line[3])), line[4], int(line[10]), level=level)
        print("end all files")
        shutil.make_archive(out_path, 'zip', out_path)
        shutil.rmtree(out_path)
        return

    def run(self, in_path, out_path):
        self.construct(in_path, out_path, level=10)


if __name__ == "__main__":
    import time
    t1 = time.time()
    in_list = r"D:\Memphis\FIR8725\test_file\pre_lob_20150102"
    out_list = r"D:\Memphis\FIR8725\test_file\out_lob_20150102"
    construct = ConstructLOB()
    construct.run(in_list, out_list)
    print("running time:", time.time() - t1)
