from pyspark.sql import SparkSession
import os
from pyspark.sql.types import IntegerType, ShortType, ByteType, StringType, DoubleType, LongType
from pyspark.sql.functions import lit, when, udf, avg
import datetime
import matplotlib.pyplot as plt


spark = SparkSession.builder.master("local[10]").appName("ITCHSummary").getOrCreate()


def read_a_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "A_*.csv"))


def read_b_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "B_*.csv"))


def read_c_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "C_*.csv"))


def read_d_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "D_*.csv"))


def read_e_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "E_*.csv"))


def read_f_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "F_*.csv"))


def read_h_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "H_*.csv"))


def read_smallh_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "smallh_*.csv"))


def read_i_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "I_*.csv"))


def read_j_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "J_*.csv"))


def read_k_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "K_*.csv"))


def read_l_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "L_*.csv"))


def read_p_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "P_*.csv"))


def read_q_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "Q_*.csv"))


def read_r_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "R_*.csv"))


def read_s_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "S_*.csv"))


def read_u_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "U_*.csv"))


def read_v_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "V_*.csv"))


def read_w_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "W_*.csv"))


def read_x_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "X_*.csv"))


def read_y_df():
    return spark.read.format('csv') \
          .option('header', 'true') \
          .load(os.path.join(os.getcwd(), "Y_*.csv"))


def create_interval(l=15):
    """
    :param l: interval length in minutes
    :return: upper bound and lower bound in nano-seconds for intervals during 9:30 to 16:00
    """
    return [((34200 + i * l * 60) * 1000000000, (34200 + (i + 1) * l * 60) * 1000000000) for i in range(int(390 / l))]


def create_when_expr(intervals, df_name):
    """

    :param intervals: minutes-based intervals with upper bound and lower bound
    :param df_name: dataframe name to use the returned when expr
    :return: an when expression which will be passed to pyspark df
    """
    return (".".join(["when( ({0}['timestamp'] > {1}) & ({0}['timestamp'] <= {2}), {3})"
                    .format(df_name, intervals[x][0], intervals[x][1], x + 1) for x in range(0, len(intervals))]) + \
                    ".otherwise(0)")


if __name__ == "__main__":
    data_dir = r"C:\Users\lguo5\itch_seminar"
    file = "S010215-v50"
    os.chdir(os.path.join(data_dir, file))
    a_df = read_a_df()
    a_df = a_df.withColumn("msg_type", lit("A"))
    b_df = read_b_df()
    b_df = b_df.withColumn("msg_type", lit("B"))
    c_df = read_c_df()
    c_df = c_df.withColumn("msg_type", lit("C"))
    d_df = read_d_df()
    d_df = d_df.withColumn("msg_type", lit("D"))
    e_df = read_e_df()
    e_df = e_df.withColumn("msg_type", lit("E"))
    f_df = read_f_df()
    f_df = f_df.withColumn("msg_type", lit("F"))
    h_df = read_h_df()
    h_df = h_df.withColumn("msg_type", lit("H"))
    smallh_df = read_smallh_df()
    i_df = read_i_df()
    i_df = i_df.withColumn("msg_type", lit("I"))
    j_df = read_j_df()
    j_df = j_df.withColumn("msg_type", lit("J"))
    k_df = read_k_df()
    k_df = k_df.withColumn("msg_type", lit("K"))
    l_df = read_l_df()
    l_df = l_df.withColumn("msg_type", lit("L"))
    p_df = read_p_df()
    p_df = p_df.withColumn("msg_type", lit("P"))
    q_df = read_q_df()
    q_df = q_df.withColumn("msg_type", lit("Q"))
    r_df = read_r_df()
    r_df = r_df.withColumn("msg_type", lit("R"))
    s_df = read_s_df()
    s_df = s_df.withColumn("msg_type", lit("S"))
    u_df = read_u_df()
    u_df = u_df.withColumn("msg_type", lit("U"))
    v_df = read_v_df()
    v_df = v_df.withColumn("msg_type", lit("V"))
    w_df = read_w_df()
    w_df = w_df.withColumn("msg_type", lit("W"))
    x_df = read_x_df()
    x_df = x_df.withColumn("msg_type", lit("X"))
    y_df = read_y_df()
    y_df = y_df.withColumn("msg_type", lit("Y"))

    merged_df = a_df.unionByName(f_df, allowMissingColumns=True).unionByName(e_df, allowMissingColumns=True) \
                    .unionByName(c_df, allowMissingColumns=True).unionByName(x_df, allowMissingColumns=True) \
                    .unionByName(d_df, allowMissingColumns=True).unionByName(u_df, allowMissingColumns=True)
    merged_df = merged_df.withColumn("stock_locate", merged_df["stock_locate"].cast(ShortType())) \
                         .withColumn("timestamp", merged_df["timestamp"].cast(DoubleType())) \
                         .withColumn("tracking_num", merged_df["tracking_num"].cast(ShortType())) \
                         .withColumn("order_ref_num", merged_df["order_ref_num"].cast(LongType())) \
                         .withColumn("shares", merged_df["shares"].cast(IntegerType())) \
                         .withColumn("price", merged_df["price"].cast(IntegerType())) \
                         .withColumn("executed_shares", merged_df["executed_shares"].cast(IntegerType())) \
                         .withColumn("match_num", merged_df["match_num"].cast(LongType())) \
                         .withColumn("execution_price", merged_df["execution_price"].cast(IntegerType())) \
                         .withColumn("canceled_shares", merged_df["canceled_shares"].cast(IntegerType())) \
                         .withColumn("ori_order_ref_num", merged_df["ori_order_ref_num"].cast(LongType())) \
                         .withColumn("new_order_ref_num", merged_df["new_order_ref_num"].cast(LongType()))

    ordered_df = merged_df.select("msg_type", "stock_locate", "tracking_num",	"timestamp",
                                  "order_ref_num", "buy_sell_indicator", "shares",
                                  "ticker", "price", "attribution", "executed_shares",
                                  "match_num", "printable", "execution_price", "canceled_shares",
                                  "ori_order_ref_num", "new_order_ref_num")\
                          .orderBy("stock_locate", "timestamp", "tracking_num")

    l = create_interval(5)
    ordered_df = ordered_df.withColumn("IntervalIndicator", eval(create_when_expr(l, "ordered_df")))
    act = ordered_df.groupBy("msg_type", "stock_locate", "IntervalIndicator").count().filter("IntervalIndicator > 0")\
                    .groupBy("msg_type", "IntervalIndicator").agg(avg("count").alias("Avg_num_order"))\
                    .orderBy("msg_type", "IntervalIndicator")
    act_df = act.toPandas()
    d = datetime.datetime(2015, 1, 2, 9, 30)
    mapper = {i: d + datetime.timedelta(minutes=i * 5) for i in range(1, len(l) + 1)}
    act_df["time"] = act_df['IntervalIndicator'].map(mapper)
    act_df.set_index("time", inplace=True)
    act_df.groupby("msg_type")["Avg_num_order"].plot(legend=True)
    plt.show()
    #ordered_df.write.csv(r"D:\Memphis\FIR8725\test_file\pre_lob_20200106")
