import sys
from parse import PreProcessing
from model import Seq2Seq
import tensorflow as tf
file_path = './chat_file/the_goblin.txt'
pps = PreProcessing(file_path)
s2s = Seq2Seq(pps.vocaes_len)


if __name__ == "__main__":
    with tf.Session() as s:
        while True:
            print(">", flush=True, end='')
            line = sys.stdin.readline()
            if(line.strip() == 'quit'):
                break
            query = pps.comm2vec(line.strip())

            chkp_sts = tf.train.get_checkpoint_state('./model')
            s2s.saver.restore(s, chkp_sts.model_checkpoint_path)
            cur_idx = 0
            de_input = []
            for i in range(40):
                a, b = pps.list2mats2(query, de_input)
                outputs = s2s.predict(s, a, b)
                if outputs[0][cur_idx] == 2:
                    break
                elif outputs[0][cur_idx] not in [0,1,2]:
                    de_input.append(outputs[0][cur_idx])
                    cur_idx += 1
            print(' '.join([pps.i2w(i) for i in de_input]))
            