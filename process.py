import struct
import pandas as pd
import gzip

def get_msg_size(binary_str):
    return struct.unpack(">H", binary_str)[0]


def get_msg_type(binary_str):
    return binary_str.decode("ascii")


def decode_timestamp(timestamp):
    new_bytes = struct.pack('>2s6s', b'\x00\x00', timestamp)  # Add padding bytes
    new_timestamp = struct.unpack('>Q', new_bytes)
    return new_timestamp[0]


def get_version5_s(binary_str):
    return list(struct.unpack(">HH6sc", binary_str))


def convert_s_2pd(S):
    df = pd.DataFrame(S, columns=["stock_locate", "tracking_num", "timestamp", "event_code"])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    return df


def get_version5_r(binary_str):
    return list(struct.unpack(">HH6s8sccIcc2scccccIc", binary_str))


def convert_r_2pd(R):
    df = pd.DataFrame(R, columns=["stock_locate", "tracking_num", "timestamp",
                                  "ticker", "mkt_cat", "finstats", "round_lot_size",
                                  "rounds_lot_only", "issue_class", "issue_sub_type",
                                  "authenticity", "short_sell", "ipo_flag", "luldref",
                                  "etp_flag", "etp_leverage", "inverse_indicator"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    df["ticker"] = df["ticker"].apply(lambda x: x.decode("ascii"))
    return df


def get_version5_h(binary_str):
    return list(struct.unpack(">HH6s8scc4s", binary_str))


def convert_h_2pd(H):
    df = pd.DataFrame(H, columns=["stock_locate", "tracking_num", "timestamp",
                                  "ticker", "trading_state", "reserved", "reason"])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    df["ticker"] = df["ticker"].apply(lambda x: x.decode("ascii"))
    return df


def get_version5_y(binary_str):
    return list(struct.unpack(">HH6s8sc", binary_str))


def convert_y_2pd(Y):
    df = pd.DataFrame(Y, columns=["stock_locate", "tracking_num", "timestamp",
                                  "ticker", "reg_sho_action"])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    df["ticker"] = df["ticker"].apply(lambda x: x.decode("ascii"))
    return df


def get_version5_l(binary_str):
    return list(struct.unpack(">HH6s4s8sccc", binary_str))


def convert_l_2pd(L):
    df = pd.DataFrame(L, columns=["stock_locate", "tracking_num", "timestamp",
                                  "mpid", "ticker", "primary_mm", "mm_mode", "mm_state"])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    df["ticker"] = df["ticker"].apply(lambda x: x.decode("ascii"))
    df["mpid"] = df["mpid"].apply(lambda x: x.decode("ascii"))
    return df


def get_version5_a(binary_str):
    return list(struct.unpack(">HH6sQcI8sI", binary_str))


def convert_a_2pd(A):
    df = pd.DataFrame(A, columns=["stock_locate", "tracking_num", "timestamp",
                                  "order_ref_num", "buy_sell_indicator", "shares",
                                  "ticker", "price"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    df["buy_sell_indicator"] = df["buy_sell_indicator"].apply(lambda x: x.decode("ascii"))
    df["ticker"] = df["ticker"].apply(lambda x: x.decode("ascii"))
    return df


def get_version5_d(binary_str):
    return list(struct.unpack(">HH6sQ", binary_str))


def convert_d_2pd(D):
    df = pd.DataFrame(D, columns=["stock_locate", "tracking_num", "timestamp",
                                  "order_ref_num"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    return df


def get_version5_e(binary_str):
    return list(struct.unpack(">HH6sQIQ", binary_str))


def convert_e_2pd(E):
    df = pd.DataFrame(E, columns=["stock_locate", "tracking_num", "timestamp",
                                  "order_ref_num", "executed_shares", "match_num"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    return df


def get_version5_f(binary_str):
    return list(struct.unpack(">HH6sQcI8sI4s", binary_str))


def convert_f_2pd(F):
    df = pd.DataFrame(F, columns=["stock_locate", "tracking_num", "timestamp",
                                  "order_ref_num", "buy_sell_indicator", "shares",
                                  "ticker", "price", "attribution"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    df["buy_sell_indicator"] = df["buy_sell_indicator"].apply(lambda x: x.decode("ascii"))
    df["ticker"] = df["ticker"].apply(lambda x: x.decode("ascii"))
    df["attribution"] = df["attribution"].apply(lambda x: x.decode("ascii"))
    return df


def get_version5_u(binary_str):
    return list(struct.unpack(">HH6sQQII", binary_str))


def convert_u_2pd(U):
    df = pd.DataFrame(U, columns=["stock_locate", "tracking_num", "timestamp",
                                  "ori_order_ref_num", "new_order_ref_num", "shares", "price"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    return df


def get_version5_x(binary_str):
    return list(struct.unpack(">HH6sQI", binary_str))


def convert_x_2pd(X):
    df = pd.DataFrame(X, columns=["stock_locate", "tracking_num", "timestamp",
                                  "order_ref_num", "canceled_shares"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    return df


def get_version5_p(binary_str):
    return list(struct.unpack(">HH6sQcI8sIQ", binary_str))


def convert_p_2pd(P):
    df = pd.DataFrame(P, columns=["stock_locate", "tracking_num", "timestamp",
                                  "order_ref_num", "buy_sell_indicator", "shares",
                                  "ticker", "price", "match_num"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    df["buy_sell_indicator"] = df["buy_sell_indicator"].apply(lambda x: x.decode("ascii"))
    df["ticker"] = df["ticker"].apply(lambda x: x.decode("ascii"))
    return df


def get_version5_k(binary_str):
    return list(struct.unpack(">HH6s8sIcI", binary_str))


def convert_k_2pd(K):
    df = pd.DataFrame(K, columns=["stock_locate", "tracking_num", "timestamp",
                                  "ticker", "ipo_quotation_re_time", "ipo_quotation_re_qu", "ipo_price"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    df["ticker"] = df["ticker"].apply(lambda x: x.decode("ascii"))
    return df


def get_version5_v(binary_str):
    return list(struct.unpack(">HH6sQQQ", binary_str))


def convert_v_2pd(V):
    df = pd.DataFrame(V, columns=["stock_locate", "tracking_num", "timestamp",
                                  "level1", "level2", "level3"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    return df


def get_version5_c(binary_str):
    return list(struct.unpack(">HH6sQIQcI", binary_str))


def convert_c_2pd(C):
    df = pd.DataFrame(C, columns=["stock_locate", "tracking_num", "timestamp",
                                  "order_ref_num", "executed_shares", "match_num",
                                  "printable", "execution_price"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    return df


def get_version5_i(binary_str):
    return list(struct.unpack(">HH6sQQc8sIIIcc", binary_str))


def convert_i_2pd(I):
    df = pd.DataFrame(I, columns=["stock_locate", "tracking_num", "timestamp",
                                  "paired_shares", "imbalance_shares", "imbalance_di",
                                  "ticker", "far_price", "near_price", "curr_ref_price",
                                  "cross_type", "price_var"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    df["ticker"] = df["ticker"].apply(lambda x: x.decode("ascii"))
    return df


def get_version5_q(binary_str):
    return list(struct.unpack(">HH6sQ8sIQc", binary_str))


def convert_q_2pd(Q):
    df = pd.DataFrame(Q, columns=["stock_locate", "tracking_num", "timestamp",
                                  "shares", "ticker", "cross_price",
                                  "match_num", "cross_type"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    df["ticker"] = df["ticker"].apply(lambda x: x.decode("ascii"))
    return df


def get_version5_j(binary_str):
    return list(struct.unpack(">HH6s8sIIII", binary_str))


def convert_j_2pd(J):
    df = pd.DataFrame(J, columns=["stock_locate", "tracking_num", "timestamp",
                                  "ticker", "auction_collar_ref_price", "upper_auction_price",
                                  "lower_auction_price", "auction_collar_ext"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    df["ticker"] = df["ticker"].apply(lambda x: x.decode("ascii"))
    return df


def get_version5_b(binary_str):
    return list(struct.unpack(">HH6sQ", binary_str))


def convert_b_2pd(B):
    df = pd.DataFrame(B, columns=["stock_locate", "tracking_num", "timestamp",
                                  "match_num"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    return df


def get_version5_smallh(binary_str):
    return list(struct.unpack(">HH6s8scc", binary_str))


def convert_smallh_2pd(h):
    df = pd.DataFrame(h, columns=["stock_locate", "tracking_num", "timestamp",
                                  "ticker", "market_code", "oha"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    df["ticker"] = df["ticker"].apply(lambda x: x.decode("ascii"))
    return df


def get_version5_w(binary_str):
    return list(struct.unpack(">HH6sc", binary_str))


def convert_w_2pd(W):
    df = pd.DataFrame(W, columns=["stock_locate", "tracking_num", "timestamp",
                                  "breached_level"
                                  ])
    df["timestamp"] = df["timestamp"].apply(lambda x: decode_timestamp(x))
    return df


def read_raw(path, version="5.0"):
    file_name = os.path.basename(path).split(".")[0]
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    os.chdir(os.path.join(os.getcwd(), file_name))
    print(os.getcwd())
    with gzip.open(path, 'rb') as file:
        reading = True
        S = []
        R = []
        H = []
        Y = []
        L = []
        A = []
        D = []
        E = []
        F = []
        U = []
        X = []
        P = []
        K = []
        V = []
        C = []
        I = []
        Q = []
        J = []
        B = []
        h = []
        W = []
        count = 0
        file_num = 1
        if version == "5.0":
            while reading:
                count += 1
                # print("Add order", len(A), "MM", len(L), "Exec", len(E))
                try:
                    msg_size = get_msg_size(file.read(2))
                except struct.error:
                    print("end of the msg")
                    msg_size = None
                msg_type = get_msg_type(file.read(1))
                if msg_type == "S":
                    S.append(get_version5_s(file.read(msg_size - 1)))
                elif msg_type == "R":
                    R.append(get_version5_r(file.read(msg_size - 1)))
                elif msg_type == "H":
                    H.append(get_version5_h(file.read(msg_size - 1)))
                elif msg_type == "Y":
                    Y.append(get_version5_y(file.read(msg_size - 1)))
                elif msg_type == "L":
                    L.append(get_version5_l(file.read(msg_size - 1)))
                elif msg_type == "A":
                    A.append(get_version5_a(file.read(msg_size - 1)))
                elif msg_type == "D":
                    D.append(get_version5_d(file.read(msg_size - 1)))
                elif msg_type == "E":
                    E.append(get_version5_e(file.read(msg_size - 1)))
                elif msg_type == "F":
                    F.append(get_version5_f(file.read(msg_size - 1)))
                elif msg_type == "U":
                    U.append(get_version5_u(file.read(msg_size - 1)))
                elif msg_type == "X":
                    X.append(get_version5_x(file.read(msg_size - 1)))
                elif msg_type == "P":
                    P.append(get_version5_p(file.read(msg_size - 1)))
                elif msg_type == "K":
                    K.append(get_version5_k(file.read(msg_size - 1)))
                elif msg_type == "V":
                    V.append(get_version5_v(file.read(msg_size - 1)))
                elif msg_type == "C":
                    C.append(get_version5_c(file.read(msg_size - 1)))
                elif msg_type == "I":
                    I.append(get_version5_i(file.read(msg_size - 1)))
                elif msg_type == "Q":
                    Q.append(get_version5_q(file.read(msg_size - 1)))
                elif msg_type == "J":
                    J.append(get_version5_j(file.read(msg_size - 1)))
                elif msg_type == "B":
                    B.append(get_version5_b(file.read(msg_size - 1)))
                elif msg_type == "h":
                    h.append(get_version5_smallh(file.read(msg_size - 1)))
                elif msg_type == "W":
                    W.append(get_version5_w(file.read(msg_size - 1)))
                else:
                    reading = False

                if count == 20000000 or reading is False:
                    convert_s_2pd(S).to_csv("S_{}.csv".format(file_num), index=False)
                    S.clear()
                    convert_r_2pd(R).to_csv("R_{}.csv".format(file_num), index=False)
                    R.clear()
                    convert_h_2pd(H).to_csv("H_{}.csv".format(file_num), index=False)
                    H.clear()
                    convert_y_2pd(Y).to_csv("Y_{}.csv".format(file_num), index=False)
                    Y.clear()
                    convert_l_2pd(L).to_csv("L_{}.csv".format(file_num), index=False)
                    L.clear()
                    convert_a_2pd(A).to_csv("A_{}.csv".format(file_num), index=False)
                    A.clear()
                    convert_d_2pd(D).to_csv("D_{}.csv".format(file_num), index=False)
                    D.clear()
                    convert_e_2pd(E).to_csv("E_{}.csv".format(file_num), index=False)
                    E.clear()
                    convert_f_2pd(F).to_csv("F_{}.csv".format(file_num), index=False)
                    F.clear()
                    convert_u_2pd(U).to_csv("U_{}.csv".format(file_num), index=False)
                    U.clear()
                    convert_x_2pd(X).to_csv("X_{}.csv".format(file_num), index=False)
                    X.clear()
                    convert_p_2pd(P).to_csv("P_{}.csv".format(file_num), index=False)
                    P.clear()
                    convert_k_2pd(K).to_csv("K_{}.csv".format(file_num), index=False)
                    K.clear()
                    convert_v_2pd(V).to_csv("V_{}.csv".format(file_num), index=False)
                    V.clear()
                    convert_c_2pd(C).to_csv("C_{}.csv".format(file_num), index=False)
                    C.clear()
                    convert_i_2pd(I).to_csv("I_{}.csv".format(file_num), index=False)
                    I.clear()
                    convert_q_2pd(Q).to_csv("Q_{}.csv".format(file_num), index=False)
                    Q.clear()
                    convert_j_2pd(J).to_csv("J_{}.csv".format(file_num), index=False)
                    J.clear()
                    convert_b_2pd(B).to_csv("B_{}.csv".format(file_num), index=False)
                    B.clear()
                    convert_smallh_2pd(h).to_csv("smallh_{}.csv".format(file_num), index=False)
                    h.clear()
                    convert_w_2pd(W).to_csv("W_{}.csv".format(file_num), index=False)
                    W.clear()
                    file_num += 1
                    count = 0

    return None


if __name__ == "__main__":
    import time
    import os
    os.chdir(r"C:\Users\lguo5\itch_seminar\output_sample")
    t1 = time.time()
    path = r"C:\Users\lguo5\itch_seminar\S010215-v50.txt.gz"
    read_raw(path)
    t2 = time.time()
    print(t2 - t1)
