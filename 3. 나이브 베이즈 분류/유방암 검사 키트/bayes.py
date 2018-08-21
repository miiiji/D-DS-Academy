def main():
    sensitivity = 0.8 #암일 떄 양성일 확률
    prior_prob = 0.004 #실제로 유방암을 가지고 있을 확률
    false_alarm = 0.1 #유방암 아닐때 양성일 확률
    
    print("%.2lf%%" % (100 * mammogram_test(sensitivity, prior_prob, false_alarm)))

def mammogram_test(sensitivity, prior_prob, false_alarm):
    p_a1_b1 = sensitivity # p(A = 1 | B = 1) # 유방암을 가질때 암이라고 진단 될 학률
    
    p_b1 = prior_prob# p(B = 1)
    
    p_b0 = 1 - p_b1# p(B = 0)
    
    p_a1_b0 = false_alarm # p(A = 1|B = 0)
    
    # P(A = 1) = P(A = 1 | B = 0)P(B = 0) + P(A=1|B=1)P(B=1)
    p_a1 = p_a1_b0 * p_b0 + p_b1*p_a1_b1 # p(A = 1)
    
    p_b1_a1 = p_a1_b1*p_b1 / p_a1# p(B = 1|A = 1)
    
    return p_b1_a1

if __name__ == "__main__":
    main()

