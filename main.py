# coding:utf8
from util import config
from random import shuffle, choice
from util.relation import Relation
from util.config import Config
from util.io import FileIO
from scipy.sparse import coo_matrix
import tensorflow as tf
import numpy as np


class HGNNLDA:
    def __init__(self, conf, trainingSet=None, testSet=None):
        self.config = conf
        self.data = Relation(self.config, trainingSet, testSet)
        self.num_lncRNAs, self.num_drugs, self.train_size = self.data.trainingSize()
        self.emb_size = int(self.config['num.factors'])
        self.maxIter = int(self.config['num.max.iter'])
        learningRate = config.LineConfig(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regU, self.regI, self.regB = float(regular['-u']), float(regular['-i']), float(regular['-b'])
        self.batch_size = int(self.config['batch_size'])
        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")
        self.r = tf.placeholder(tf.float32, name="rating")
        self.lncRNA_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_lncRNAs, self.emb_size], stddev=0.005),
                                           name='U')
        self.drug_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_drugs, self.emb_size], stddev=0.005),
                                           name='V')
        self.u_embedding = tf.nn.embedding_lookup(self.lncRNA_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.drug_embeddings, self.v_idx)
        config1 = tf.ConfigProto()
        config1.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config1)
        self.loss, self.lastLoss = 0, 0

    def buildAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            row += [self.data.lncRNA[pair[0]]]
            col += [self.data.drug[pair[1]]]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.num_lncRNAs, self.num_drugs), dtype=np.float32)
        return u_i_adj

    def initModel(self):
        A = self.buildAdjacencyMatrix()
        H_u = A
        D_u_v = H_u.sum(axis=1).reshape(1, -1)
        D_u_e = H_u.sum(axis=0).reshape(1, -1)
        temp1 = (H_u.transpose().multiply(np.sqrt(1.0 / D_u_v))).transpose()
        temp2 = temp1.transpose()
        A_u = temp1.multiply(1.0 / D_u_e).dot(temp2)
        A_u = A_u.tocoo()
        indices = np.mat([A_u.row, A_u.col]).transpose()
        H_u = tf.SparseTensor(indices, A_u.data.astype(np.float32), A_u.shape)
        H_i = A.transpose()
        D_i_v = H_i.sum(axis=1).reshape(1, -1)
        D_i_e = H_i.sum(axis=0).reshape(1, -1)
        temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()
        temp2 = temp1.transpose()
        A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)
        A_i = A_i.tocoo()
        indices = np.mat([A_i.row, A_i.col]).transpose()
        H_i = tf.SparseTensor(indices, A_i.data.astype(np.float32), A_i.shape)
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        initializer = tf.contrib.layers.xavier_initializer()
        self.n_layer = 3
        self.weights = {}
        for i in range(self.n_layer):
            self.weights['layer_%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]),
                                                             name='JU_%d' % (i + 1))

        lncRNA_embeddings = self.lncRNA_embeddings
        drug_embeddings = self.drug_embeddings
        all_lncRNA_embeddings = [lncRNA_embeddings]
        all_drug_embeddings = [drug_embeddings]

        def without_dropout(embedding):
            return embedding

        def dropout(embedding):
            return tf.nn.dropout(embedding, rate=0.1)

        for i in range(self.n_layer):
            new_lncRNA_embeddings = tf.sparse_tensor_dense_matmul(H_u, self.lncRNA_embeddings)
            new_drug_embeddings = tf.sparse_tensor_dense_matmul(H_i, self.drug_embeddings)

            lncRNA_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_lncRNA_embeddings, self.weights['layer_%d' % (i + 1)]) + lncRNA_embeddings)
            drug_embeddings = tf.nn.leaky_relu(
                tf.matmul(new_drug_embeddings, self.weights['layer_%d' % (i + 1)]) + drug_embeddings)

            lncRNA_embeddings = tf.cond(self.isTraining, lambda: dropout(lncRNA_embeddings),
                                      lambda: without_dropout(lncRNA_embeddings))
            drug_embeddings = tf.cond(self.isTraining, lambda: dropout(drug_embeddings),
                                      lambda: without_dropout(drug_embeddings))

            lncRNA_embeddings = tf.math.l2_normalize(lncRNA_embeddings, axis=1)
            drug_embeddings = tf.math.l2_normalize(drug_embeddings, axis=1)

            all_drug_embeddings.append(drug_embeddings)
            all_lncRNA_embeddings.append(lncRNA_embeddings)
        lncRNA_embeddings = tf.concat(all_lncRNA_embeddings, axis=1)
        drug_embeddings = tf.concat(all_drug_embeddings, axis=1)
        self.final_lncRNA = lncRNA_embeddings
        self.final_drug = drug_embeddings
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_drug_embedding = tf.nn.embedding_lookup(drug_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(lncRNA_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(drug_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding, drug_embeddings), 1)

    def next_batch_pairwise(self):
        shuffle(self.data.trainingData)
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                lncRNAs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)]
                drugs = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)]
                batch_id += self.batch_size
            else:
                lncRNAs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                drugs = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size

            u_idx, i_idx, j_idx = [], [], []
            drug_list = list(self.data.drug.keys())
            for i, lncRNA in enumerate(lncRNAs):
                i_idx.append(self.data.drug[drugs[i]])
                u_idx.append(self.data.lncRNA[lncRNA])
                neg_drug = choice(drug_list)
                while neg_drug in self.data.trainSet_u[lncRNA]:
                    neg_drug = choice(drug_list)
                j_idx.append(self.data.drug[neg_drug])

            yield u_idx, i_idx, j_idx

    def buildModel(self):
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_drug_embedding), 1)
        reg_loss = self.regU * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
                                tf.nn.l2_loss(self.neg_drug_embedding))
        for i in range(self.n_layer):
            reg_loss += self.regU * tf.nn.l2_loss(self.weights['layer_%d' % (i + 1)])
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + reg_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for iteration in range(self.maxIter):
            for n, batch in enumerate(self.next_batch_pairwise()):
                lncRNA_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                     feed_dict={self.u_idx: lncRNA_idx, self.neg_idx: j_idx, self.v_idx: i_idx,
                                                self.isTraining: 1})
                print('training:', iteration + 1, 'batch', n, 'loss:', l)

    def execute(self, i):
        self.initModel()
        print('Building Model ...')
        self.buildModel()


if __name__ == '__main__':
    conf = Config('HGNNLDA.conf')
    for i in range(0, 5):
        train_path = f"./dataset/train_{i}.txt"
        test_path = f"./dataset/test_{i}.txt"
        trainingData = FileIO.loadDataSet(conf, train_path)
        testData = FileIO.loadDataSet(conf, test_path, bTest=True)
        re = HGNNLDA(conf, trainingData, testData)
        re.execute(i)
