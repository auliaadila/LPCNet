import numpy as np
from tensorflow.keras.utils import Sequence
from ulaw import lin2ulaw

def lpc2rc(lpc):
    #print("shape is = ", lpc.shape)
    order = lpc.shape[-1]
    rc = 0*lpc
    for i in range(order, 0, -1):
        rc[:,:,i-1] = lpc[:,:,-1]
        ki = rc[:,:,i-1:i].repeat(i-1, axis=2)
        lpc = (lpc[:,:,:-1] - ki*lpc[:,:,-2::-1])/(1-ki*ki)
    return rc

BITS_PER_FRAME = 64

class LPCNetLoader(Sequence):
    def __init__(self, data, features, periods, batch_size, bits_in=None, e2e=False, lookahead=2):
        self.batch_size = batch_size
        self.nb_batches = np.minimum(np.minimum(data.shape[0], features.shape[0]), periods.shape[0])//self.batch_size
        self.data = data[:self.nb_batches*self.batch_size, :]
        self.features = features[:self.nb_batches*self.batch_size, :]
        self.periods = periods[:self.nb_batches*self.batch_size, :]
        self.e2e = e2e
        self.lookahead = lookahead
        self.bps = BITS_PER_FRAME
        self.bits_in = np.random.randint(0, 2, size=(1, self.bps), dtype='int32')

        # (bs, 2400, 64)
        # 64 bps: 64 bit per second
        # 64 bpf: 64 bit per frame

        # print("INIT LPCNET LOADER")
        # print("nb_batches:", self.nb_batches) #0
        # print("data:", self.data.shape) #data: (0, 2400, 2)
        # print("features:", self.features.shape) #features: (0, 19, 36)
        # print("periods:",self.periods.shape) #periods: (0, 19, 1)

        # copy to all batch size
        self.bits_in = np.broadcast_to(self.bits_in,(self.batch_size, self.data.shape[1], self.bps))
        # data = np.reshape(data, (nb_frames, pcm_chunk_size, 2))

        '''
        if bits_in is None:
            self._make_random_bits()        # fills self.bits_in
        else:
            assert bits_in.shape[:2] == self.data.shape[:2], \
                "bits_in must match (N,F) of data" #why????
            self.bits_in = bits_in.astype('int32')
        '''

        self.on_epoch_end()

    def _make_random_bits(self):
        N, F = self.data.shape[:2] #batch_size, nb_frames
        # print("target bit shape")
        # print(N,F) #0,19
        self.bits_in = np.random.randint(
            0, 2, size=(N, F, self.bps), dtype='int32') #bps: to be embedded per second
        
    def on_epoch_end(self):
        self.indices = np.arange(self.nb_batches*self.batch_size)
        # print("indices:",self.indices)
        np.random.shuffle(self.indices)
        # regenerate a fresh random payload each epoch (optional) -> yes
        # self._make_random_bits()


    def __getitem__(self, index):
        # print("GET ITEM")
        # print("index:",index)
        # print("id:",index*self.batch_size)
        # print("(id + 1):",(index+1)*self.batch_size)
        # per batch size
        data = self.data[self.indices[index*self.batch_size:(index+1)*self.batch_size], :, :]
        in_data = data[: , :, :1]
        out_data = data[: , :, 1:]
        features = self.features[self.indices[index*self.batch_size:(index+1)*self.batch_size], :, :-16]
        periods = self.periods[self.indices[index*self.batch_size:(index+1)*self.batch_size], :, :]
        bits_in = self.bits_in

        #generate bits_in

        # print("data:", data.shape)
        # print("in_data:", in_data.shape) #mirip bits_in
        # print("out_data:", out_data.shape)
        # print("features:", features.shape)
        # print("periods:", periods.shape)
        # print("bits_in:", bits_in.shape)

        outputs = out_data
        inputs = [in_data, features, periods, bits_in] #will add lpc
        
        if self.lookahead > 0:
            lpc = self.features[self.indices[index*self.batch_size:(index+1)*self.batch_size], 4-self.lookahead:-self.lookahead, -16:]
        else:
            lpc = self.features[self.indices[index*self.batch_size:(index+1)*self.batch_size], 4:, -16:]
        if self.e2e:
            outputs.append(lpc2rc(lpc))
        else:
            inputs.append(lpc)
        
        # print("Inputs shapes:")
        # for i, inp in enumerate(inputs):
        #     print(f"Input {i}: {inp.shape}")

        # print("Outputs shapes:")
        # for i, out in enumerate(outputs):
        #     print(f"Output {i}: {out.shape}")

        # Inputs shapes:
        # Input 0: (128, 2400, 1)
        # Input 1: (128, 19, 20)
        # Input 2: (128, 19, 1)
        # Input 3: (128, 2400, 64)
        # Input 4: (128, 15, 16)
        # Outputs shapes:
        # Output 0: (2400, 1)

        return (inputs, outputs)
        

    def __len__(self):
        return self.nb_batches


# INIT LPCNET LOADER
# nb_batches: 0
# data: (0, 2400, 2)
# features: (0, 19, 36)
# periods: (0, 19, 1)
# indices: []
# GET ITEM
# index: 0
# id: 0
# (id + 1): 128
# data: (0, 2400, 2)
# in_data: (0, 2400, 1)
# out_data: (0, 2400, 1)
# features: (0, 19, 20)
# periods: (0, 19, 1)
# bits_in: (0, 19, 64)