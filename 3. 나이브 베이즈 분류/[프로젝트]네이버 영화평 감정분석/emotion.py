import io
import numpy
import csv
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import elice_utils

special_chars_remover = re.compile("[^\w'|_]")
def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)

def main():
    training_sentences = read_data()
    testing_sentence = "어설픈 연기들로 몰입이 전혀 안되네요"
    prob_pair = naive_bayes(training_sentences, testing_sentence)
    
    plot_title = testing_sentence
    if len(plot_title) > 50: plot_title = plot_title[:50] + "..."
    visualize_boxplot(plot_title,
                      list(prob_pair),
                      ['Negative', 'Positive'])

def naive_bayes(training_sentences, testing_sentence):
    log_prob_negative = calculate_doc_prob(training_sentences[0], testing_sentence, 0.1) + math.log(0.5)
    log_prob_positive = calculate_doc_prob(training_sentences[1], testing_sentence, 0.1) + math.log(0.5)
    prob_pair = normalize_log_prob(log_prob_negative, log_prob_positive)
    
    return prob_pair

def read_data():
    training_sentences = [[], []]
    # words.txt 에서 단어들를 읽어,
    # [[단어1, 빈도수], [단어2, 빈도수] ... ]형으로 변환해 리턴합니다.
    csvreader = csv.reader(open("./ratings.txt"),delimiter='\t');
    next(csvreader)
    for line in csvreader:
        if(float(line[2]) == 0):
            training_sentences[0].append(line[1])
        elif(float(line[2]) == 1):
            training_sentences[1].append(line[1])

'''
    문제 1
    여기서 파일을 읽어 training_sentences에 저장합니다.
    '''
        return [' '.join(training_sentences[0]), ' '.join(training_sentences[1])]

def normalize_log_prob(prob1, prob2):
    
    '''
        문제 4
        로그로 된 확률값을 표준화합니다.
        이 부분은 이미 작성되어 있습니다.
        '''
    
    maxprob = max(prob1, prob2)
    
    prob1 -= maxprob
    prob2 -= maxprob
    prob1 = math.exp(prob1)
    prob2 = math.exp(prob2)
    
    normalize_constant = 1.0 / float(prob1 + prob2)
    prob1 *= normalize_constant
    prob2 *= normalize_constant
    
    return (prob1, prob2)

def calculate_doc_prob(training_sentence, testing_sentence, alpha):
    logprob = 0
    training_model = create_BOW(training_sentence) #training을 bow로 만듬
    testing_model = create_BOW(testing_sentence) #testing을 bow로 만듬
    #trainig_list = []
    total = 0
    total_test = 0
    
    for value in training_model.values():
        total += int(value)
    for key,value in training_model.items():
        training_model[key] = float(value/total)

    for key, value in testing_model.items():
        if key in training_model:
            logprob += math.log(training_model[key]) * value;
        else:
            logprob += (math.log(alpha) - math.log( total))*value;
'''
    문제 3
    training_sentence로 만들어진 모델이,
    testing_sentence를 만들어 낼 **로그 확률** 을 구합니다.
    일반 숫자에서 로그값을 만들기 위해서는 math.log() 를 사용합니다.
    
    일반 숫자에서의 곱셈이 로그에서는 덧셈, 나눗셈은 뺄셈이 된다는 점에 유의하세요.
    예) 3 * 5 = 15
    log(3) + log(5) = log(15)
    
    5 / 2 = 2.5
    log(5) - log(2) = log(2.5)
    '''
        print(logprob)
        return logprob

def create_BOW(sentence):
    bow = {}
    sentence_lowered = sentence.lower()
    sentence_without_s_c = remove_special_characters(sentence_lowered)
    splitted_sentence = sentence_without_s_c.split()
    
    splitted_sentence_filtered = [
                                  token
                                  for token in splitted_sentence
                                  if len(token) >= 1
                                  ]
        
                                  for token in splitted_sentence_filtered:
                                      # if token not in bow:
                                      #     bow[token] = 1
                                      # else:
                                      #     bow[token] += 1
                                      # 아래랑 같은 코드
                                      bow.setdefault(token,0) #만약 token이 없으면 0으로 정해라
                                      bow[token]+=1 # 그리고 거기다가 1을 더해줘라

'''
    문제 2
    이전 실습과 동일하게 bag of words를 만듭니다.
    '''
        
        return bow

'''
    이 밑의 코드는 시각화를 위한 코드입니다.
    궁금하다면 살펴보세요.
    '''
def visualize_boxplot(title, values, labels):
    width = .35
    
    print(title)
    
    fig, ax = plt.subplots()
    ind = numpy.arange(len(values))
    rects = ax.bar(ind, values, width)
    ax.bar(ind, values, width=width)
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(labels)
    
    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., height + 0.01, '%.2lf%%' % (height * 100), ha='center', va='bottom')

autolabel(rects)

plt.savefig("image.svg", format="svg")
elice_utils.send_image("image.svg")

if __name__ == "__main__":
    main()

