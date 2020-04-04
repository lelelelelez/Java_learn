import numpy as np
import paddle.fluid as fluid
dict_size = 20
data_t = fluid.layers.data(name='word', shape=[1], dtype='int64', lod_level=1)
padding_idx = np.random.randint(1, 10)

##正确case
out = fluid.contrib.fused_embedding_seq_pool(
    input=data_t,
    size=[dict_size, 32],
    param_attr='w',
    padding_idx=padding_idx,
    is_sparse=False)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# prepare input words' idx
x_tensor = fluid.core.LoDTensor()
idxs = np.random.randint(1, 10, (8)).astype("int64")

x_tensor.set(idxs, place)
x_tensor.set_recursive_sequence_lengths([[4, 4]])
ret = exe.run(feed={'word': x_tensor}, fetch_list=[out])



#test1: NotFoundError: No Input(Ids) found for FusedEmbeddingSeqPool operator.
    data_t = '1'

#test2: table_dims.size()==2
    out = fluid.contrib.fused_embedding_seq_pool(
        input=data_t,
        size=[dict_size], #size=1
        param_attr='w',
        padding_idx=padding_idx,
        is_sparse=False) 

#test3: 输入的id的维度大于1
    #没有构造出来
 
#test4: The last dimension of the input tensor 'Ids' should be 1
    data_t = fluid.layers.data(name='word', shape=[2], dtype='int64', lod_level=1)

#test5: combiner 必须是sum 
    out = fluid.contrib.fused_embedding_seq_pool(
        input=data_t,
        size=[dict_size, 32],
        param_attr='w',
        padding_idx=padding_idx,
        combiner='div',
        is_sparse=False) 

#test6: In compile time, the LoD Level of Ids should be 1. But received the LoD Level of Ids = 2
    data_t = fluid.layers.data(name='word', shape=[1], dtype='int64', lod_level=2)

#test7: table_width * idx_width 小于等于 out_width
    #没有构造出来

#test8:  Ids's lod[0] 大于0
    #没有构造出来

#test9: The LoD level of Input(Ids) should be 1
    ret = exe.run(feed={'testtt': x_tensor}, fetch_list=[out]) ##testtt没有声明




