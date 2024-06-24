from pyhtm.pyhtm import ScalarEncoder, Regressor, SpatialPooler, TemporalMemory, plot_SDR

class HTM():
    def __init__(self, minval=0, maxval=100, w=5, n=100):     
        self.encoder = ScalarEncoder(n=n,w=w,minval=minval,maxval=maxval)
    
    def reset(self):
        self.tm.reset()
    
    def init_sp(self, config=None):
        if config is None:
            column_num = 100
            active_cols = 4
            perm_inc = 0.04
            perm_dec = 0.008
            perm_thresh = 0.1
            boost = 3
        else:
            column_num = config['column_num']
            active_cols = config['active_cols']
            perm_inc = config['perm_inc']
            perm_dec = config['perm_dec']
            perm_thresh = config['perm_thresh']
            boost = config['boost']
        self.sp = SpatialPooler(source=self.encoder, column_num=column_num,
                               max_active_cols=active_cols,
                               perm_increment=perm_inc,
                               perm_decrement=perm_dec,
                               boost_str=boost
                               )
    def init_tm(self, num_cells=2, stimulus_thresh=4):
        self.tm = TemporalMemory(spatial_pooler=self.sp, num_cells=num_cells, stimulus_thresh=stimulus_thresh)
        
        
    def forward_sp(self, x):
        out = self.sp.process_input(self.encoder.encode(x))
        return out
    
    def forward_tm(self, x):
        act, pred = self.tm.process_input(self.encoder.encode(x), tm_learning = True, sp_learning = False)
        return act, pred
        
    def train_reg(self, iters):
        sdrlist = []
        valuelist = []
        for e in tqdm(range(iters)):
            for i in range(vocab_size):
                valuelist.append(i)
                _, sdr = self.tm.process_input(self.encoder.encode(i))
                sdrlist.append(sdr)

        self.reg = Regressor(sdrlist, valuelist, regressor_type = 'linear')
    
    def forward(self, x):
        act, pred = self.tm.process_input(self.encoder.encode(x), tm_learning = False, sp_learning = False)
        pred = self.reg.translate(pred)
        return pred
