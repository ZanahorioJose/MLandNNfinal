from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import os

max_features = 5000
max_document_length = 100

# 将每个文件处理成为字符串的形式.

def load_one_file(filename):
    res = ""
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            res += line
    return res

# 遍历6个文件夹并加载数据.
def load_files_from_dir(rootdir):
    res = []
    list_ = os.listdir(rootdir)
    for i in range(0, len(list_)):
        path_ = os.path.join(rootdir, list_[i])
        if os.path.isfile(path_):
            v = load_one_file(path_)
            res.append(v)
    return res

# 加载所有文件. 将正常邮件与垃圾邮件分别存储与ham与spam.
def load_all_files():
    ham = []
    spam = []
    for i in range(1,7):
        path_ham = "F:/复旦大学/课程（部分）/研一秋/机器学习与神经网络导论/期末试题/Q9_spam/data/enron%d/ham" % i
        print("Loading %s" % path_ham)
        ham += load_files_from_dir(path_ham)

        path_spam = "F:/复旦大学/课程（部分）/研一秋/机器学习与神经网络导论/期末试题/Q9_spam/data/enron%d/spam/" % i
        print("Loading %s" % path_spam)
        spam += load_files_from_dir(path_spam)
    return ham, spam

# 用词袋模型将文本向量化.
# ham 标记为0, spam 标记为1.

def get_features_by_wording():
    ham, spam = load_all_files()
    X = ham + spam
    Y = [0]*len(ham) + [1]*len(spam)
    vectorizer = CountVectorizer(decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1)
    print(vectorizer)
    X = vectorizer.fit_transform(X)
    X = X.toarray()
    return X, Y

# 构建贝叶斯模型.

def naivebaysian_wordbag(X_train, X_test, Y_train, Y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    Y_pred = gnb.predict(X_test)
    print(metrics.accuracy_score(Y_test, Y_pred))
    print(metrics.confusion_matrix(Y_test, Y_pred))


X, Y = get_features_by_wording()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
naivebaysian_wordbag(X_train, X_test, Y_train, Y_test)

#%%
