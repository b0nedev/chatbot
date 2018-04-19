import tensorflow as tf
from model import Seq2Seq
from parse import PreProcessing
from config import FLAGS
    
def train(file_path, epoch=1000):
    pps = PreProcessing(file_path)
    s2s = Seq2Seq(pps.vocaes_len)
    en_input, de_input, targets = pps.list2mats()

    with tf.Session() as s:
        chkp_sts = tf.train.get_checkpoint_state('./model')
        if chkp_sts and tf.train.checkpoint_exists(chkp_sts.model_checkpoint_path):
            print('reading model...')
            s2s.saver.restore(s, chkp_sts.model_checkpoint_path)
        else:
            print('new model...')
            s.run(tf.global_variables_initializer())
        for e in range(epoch):
            _, cost = s2s.train(s, en_input, de_input, targets)
            print(cost)

        expect, outputs, accuracy = s2s.test(s, en_input, de_input, targets)
        expect = [[pps.i2w(i) for i in e] for e in expect]
        print('expect:', expect)
        print('outputs:', outputs)
        print('accuracy', accuracy)

        s2s.saver.save(s, './model/thegoblin.chkp_sts', global_step=s2s.g_step)

        
def test(file_path):
    pps = PreProcessing(file_path)
    s2s = Seq2Seq(pps.vocaes_len)
    en_input, de_input, targets = pps.list2mats()

    with tf.Session() as s:
        chkp_sts = tf.train.get_checkpoint_state('./model')
        print('reading s2s model:', chkp_sts.model_checkpoint_path)
        s2s.saver.restore(s, chkp_sts.model_checkpoint_path)
        expect, outputs, accuracy = s2s.test(s, en_input, de_input, targets)
        expect = [[pps.i2w(i) for i in e] for e in expect]
        print('expect:', expect)
        print('outputs:', outputs)
        print('accuracy', accuracy)

        
def main(_):
    if FLAGS.train:
        train('./chat_file/the_goblin.txt')
    elif FLAGS.test:
        test('./chat_file/the_goblin.txt')

        
if __name__ == '__main__':
    tf.app.run()