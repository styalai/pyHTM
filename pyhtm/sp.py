class miniColumn():
    #This object maintains a list of permanence arrays and connection arrays.
    #Each list has one array corresponding to each input source.
    #The Spatial Pooler object contains a list of these miniColumn objects.
    
    def __init__(self, own_index, potential_percent = 0.5, perm_decrement = 0.008, perm_increment = 0.05, perm_thresh = 0.1, duty_cycle_period = 1000, self_conn_str = None):
        #Constructor method.
        #own_index -> This object's index in its parent object's minicolumn list.
        #input_dim -> Expected dimensions of the input space.
        #potential_percent -> The fraction of bits in the input space to which
        #this miniColumn *may* grow connections.
        #perm_decrement -> Amount by which the permanence to an inactive bit will decrease.
        #perm_increment -> Amount by which the permanence to an active bit will increase.
        #perm_thresh -> Threshold over which a connected synapse will form.
        #duty_cycle_period -> Number of recent inputs used to compute the duty cycle.
        #self_conn_str -> The self-reinforcing connection strength, if there is one (i.e. for a TemporalPooler)
        
        #Record the numeric parameters
        self.input_dims = []
        self.potential_percent = potential_percent
        self.perm_decrement = perm_decrement
        self.perm_increment = perm_increment
        self.perm_thresh = perm_thresh
        self.duty_cycle_period = duty_cycle_period
        self.own_index = own_index
        self.self_conn_str = self_conn_str
        
        #Initialize the duty cycle tracker
        self.duty_tracker = [0]*int(duty_cycle_period)
        
        #Initialize the potential connections array list.
        #The potential connections will be stored in an array of size input_dim with 1's to indicate possible connections.
        self.potential_connections = []
        
        #Initialize the permanence array list.
        #The permanences will be initialized using a normal distribution centered on perm_thresh, multiplied pointwise by potential_connections.
        self.perms = []
        
        #Initialize the actual-connections array list.
        #The actual connections will be stored in an array like the potential connections.
        self.actual_connections = []
        
    def connect(self, input_dim):
        #Appends a new set of connection and permanence arrays to the lists,
        #matching a new input from a new source.
        
        #Append the new input dimension
        self.input_dims.append(input_dim)
        
        #Append the new potential connections array
        self.potential_connections.append(np.random.choice([1,0], size=input_dim, p=[self.potential_percent, 1-self.potential_percent]))
        
        #Append the new permanences array
        self.perms.append(np.random.normal(loc=self.perm_thresh, scale=self.perm_increment, size=input_dim)*self.potential_connections[-1])
        
        #Append the actual connections array
        self.actual_connections.append((self.perms[-1] >= self.perm_thresh).astype('int32'))
        
        #If this is a self-connection (e.g. part of a TemporalPooler), specify the
        #self-connection strength
        if self.self_conn_str is not None:
            self.potential_connections[0][self.own_index] = 1
            self.actual_connections[0][self.own_index] = self.self_conn_str
        
    def get_overlap_score(self, arr, in_src_num = 0):
        #Returns the overlap score between the actual connections and an array of active synapses.
        return np.sum(arr*self.actual_connections[in_src_num])

    def update_perms(self, arr, in_src_num = 0):
        #Increments the permanence values for active synapses.
        #Decrements the permanence values for inactive synapses.
        #Updates the actual_connections array.
        self.perms[in_src_num][(arr > 0)] += self.perm_increment
        self.perms[in_src_num][(arr < 1)] -= self.perm_decrement
        self.actual_connections[in_src_num] = self.potential_connections[in_src_num]*(self.perms[in_src_num] >= self.perm_thresh)
        
        #Maintain the self connection strength at a constant, if there is one.
        if self.self_conn_str is not None and in_src_num == 0:
            self.actual_connections[0][self.own_index] = self.self_conn_str
    
    def low_duty_cycle_inc(self, in_src_num = 0):
        #Increments all permanence values to promote an increased duty cycle.
        self.perms[in_src_num] += self.perm_increment
        self.actual_connections[in_src_num] = self.potential_connections[in_src_num]*(self.perms[in_src_num] >= self.perm_thresh)
        
        #Maintain the self connection strength at a constant, if there is one.
        if self.self_conn_str is not None and in_src_num == 0:
            self.actual_connections[0][self.own_index] = self.self_conn_str

    def duty_cycle_update(self,activity):
        #Updates the duty cycle tracker.
        self.duty_tracker.insert(0,activity) #Insert the newest activity number to the head of the list
        self.duty_tracker.pop(int(self.duty_cycle_period)) #Pop off the oldest activity number 
        
    def get_duty_cycle(self):
        #Returns the number of times within the last duty cycle period that this minicolumn activated.
        return np.sum(self.duty_tracker)
    
class SpatialPooler():
    #The SpatialPooler object curates a list of minicolumns, providing inputs and gathering outputs.
    #The pooler can maintain connections to encoders, poolers or temporal memories.
    
    def __init__(self, source = None, input_dim = None, column_num = 1000, potential_percent = 0.85, max_active_cols = 40, stimulus_thresh = 0, perm_decrement = 0.005, perm_increment = 0.04, perm_thresh = 0.1, min_duty_cycle = 0.001, duty_cycle_period = 100, boost_str = 3, ID = 'SP1'):
        #Constructor method.
        #input_source -> Connectable object that is the source of inputs for this object.
        #column_num -> Number of minicolumns to be used.
        #potential_percent -> The fraction of bits in the input space to which
        #this miniColumn *may* grow connections.
        #max_active_cols -> Number of allowed minicolumn activations per input processed
        #stimulus_thresh -> Threshold of overlap for a minicolumn to be eligible for action.
        #perm_decrement -> Amount by which the permanence to an inactive bit will decrease.
        #perm_increment -> Amount by which the permanence to an active bit will increase.
        #perm_thresh -> Threshold over which a connected synapse will form.
        #min_duty_cycle -> A minicolumn with duty cycle below this value will be encouraged to be more active.
        #duty_cycle_period -> Number of recent inputs used to compute the duty cycle.
        #boost_str -> Strength of the boosting effect used to enhance the overlap score of low-duty-cycle minicolumn.
        #ID -> Identifier for this object. Should be unique for each instance.
        
        #Record the numeric parameters
        self.input_dims = []
        self.input_sources = []
        self.column_num = column_num
        self.potential_percent = potential_percent
        self.max_active_cols = max_active_cols
        self.stimulus_thresh = stimulus_thresh
        self.perm_decrement = perm_decrement
        self.perm_increment = perm_increment
        self.perm_thresh = perm_thresh
        self.min_duty_cycle = min_duty_cycle
        self.duty_cycle_period = duty_cycle_period
        self.boost_str = boost_str
        self.output_dim = (column_num,)
        self.ID_dict = {}
        self.ID = ID
        
        #Initialize column activity tracker
        self.active_cols = np.zeros((column_num,))
        
        #Track the boost factors
        self.boost_factors = np.ones((column_num,))
                
        #Count how many inputs have been processed. Used for duty cycle tracking.
        self.input_cycles = 0
        
        ##Initialize the columns
        self.columns = [miniColumn(own_index = i, potential_percent = potential_percent, perm_decrement = perm_decrement, perm_increment = perm_increment, perm_thresh = perm_thresh, duty_cycle_period = duty_cycle_period) for i in range(column_num)]
        
        #Initialize the all_connections list.
        #The entries in this list collect all of the column connections for a given source in a single numpy array
        self.all_connections = []
        
        #If the SP constructor was called with an input source, connect to it.
        if source is not None:
            self.connect(source)
        
        if input_dim is not None:
            self.connect(DummyConnector(output_dim = input_dim))
        
    def connect(self, source):
        #Connects the SP to another object: adds another set of connections
        #to all of the minicolumns.
        self.input_dims.append(source.output_dim)
        self.input_sources.append(source.ID)
        self.ID_dict[source.ID] = len(self.input_sources)-1
        
        #Connect each column to the new input source
        for col in self.columns:
            col.connect(source.output_dim)
            
        self.all_connections.append(np.array([col.actual_connections[-1] for col in self.columns]).reshape(self.column_num,-1))
            
    def compute_overlap(self, arr, input_source_num = 0):
        #Reads in an encoded SDR, which is an array of 0s and 1s of shape input_dim that 
        #indicate inactive/active input bits respectively. Returns an output of shape 
        #(column_num,1) containing the overlap score of each minicolumn.
        
        #Get the raw overlap scores of each minicolumn.
        overlap_scores = np.zeros((self.column_num,))
        for i in range(self.column_num):
            overlap_scores[i] = self.columns[i].get_overlap_score(arr, in_src_num = input_source_num)
            
        return overlap_scores
    
    def compute_overlap_par(self, arr, input_source_num = 0):
        #Just like compute overlap, but uses the all_connections array.
        #Testing to see if this is faster.
        return np.dot(self.all_connections[input_source_num],arr)
        
    def update_boost_factors(self):
        #Recalculates the boost factors of each minicolumn if a duty cycle period has passed.
        if self.input_cycles >= self.duty_cycle_period:
            #Update the boost factors
            for i in range(self.column_num):
                self.boost_factors[i] = np.exp(-self.boost_str*(self.columns[i].get_duty_cycle()/self.duty_cycle_period - self.max_active_cols/self.column_num))
    
    def get_active_columns(self, overlaps):
        #Takes a set of overlap scores of shape (column_num,) and returns a 
        #binary array indicating inactive/active minicolumns.
        #Can be given either pre- or post-boost overlap scores.
        self.reset()
        self.active_cols[np.argpartition(overlaps,-self.max_active_cols)[-self.max_active_cols:]] = 1
        return self.active_cols
    
    def duty_cycle_update(self):
        #Updates the duty cycle data of every column.
        for i in range(self.column_num):
            self.columns[i].duty_cycle_update(self.active_cols[i])
            
    def permanence_update(self, arr, input_source_num = 0):
        #Update the permanence of active minicolumns.
        for i in range(self.column_num):
            if self.active_cols[i] > 0:
                #Update the permanences of active columns
                self.columns[i].update_perms(arr,input_source_num)
                
                #Update the all_connections array
                self.all_connections[input_source_num][i,:] = self.columns[i].actual_connections[input_source_num]
        
    def low_duty_cycle_inc(self):
        #Calls low_duty_cycle_inc() for each column with a duty cycle below the
        #minimum to encourage more activity.
        for i in range(self.column_num):
            #If the column has not been sufficiently active:
            if self.columns[i].get_duty_cycle()/self.duty_cycle_period <= self.min_duty_cycle:
                #Loop through all the input spaces
                for j in range(len(self.input_sources)):
                #Update the permanences
                    self.columns[i].low_duty_cycle_inc(in_src_num = j)
                    #Update the all_connections array
                    self.all_connections[j][i,:] = self.columns[i].actual_connections[j]
        
    def reset(self):
        #Resets the active columns
        self.active_cols = np.zeros(self.active_cols.shape)
    
    def process_input(self, arr, boosting = True, input_source_num=0, input_ID = None, new_cycle = True, sp_learning = True):
        #Takes an encoded input SDR and goes through all of the steps needed to
        #process it, I.E. determining which minicolumns become active and performing
        #the learning updates. Returns an SDR of active minicolumns.

        #The input can also be specified by ID instead of source num
        if input_ID is not None:
            input_source_num = self.ID_dict[input_ID]

        #If this is the beginning of a new input cycle:
        if new_cycle:
            #Increment the input counter. This is for duty cycle tracking purposes.
            self.input_cycles += 1     
            
            #Update the duty cycle from last time's active columns
            self.duty_cycle_update()
            
            #Periodically increment the permanences for minicolumns with low duty cycles
            if (self.input_cycles % self.duty_cycle_period == 0) and (sp_learning):
                self.low_duty_cycle_inc()
                
            #Update the boost factors
            if boosting:
                self.update_boost_factors()
            
        #Get the minicolumn overlap scores.
        overlap_scores = self.compute_overlap_par(arr, input_source_num)
        
        #Boost the overlap scores if boosting is being used.
        if boosting:
            overlap_scores = overlap_scores*self.boost_factors
            
        #Determine the active minicolumns based on the net overlap scores
        self.get_active_columns(overlap_scores);
            
        #Update the permanences for each minicolumn
        if sp_learning:
            self.permanence_update(arr,input_source_num)
        
        #Return an SDR of the active columns.
        return self.active_cols
    
    def process_multiple_inputs(self, arr_list, input_index_list = None, input_ID_list = None, boosting = True, sp_learning = True, new_cycle = True):
        #Meant to be used for instances that have multiple input sources.
        #Calculates a total overlap score and activity for columns
        #based on all inputs. Trains all connections for all input sources.
        #Returns an SDR of active minicolumns.
        
        #If input_index_list is not given, assume it is the first N sources
        if input_index_list is None:
            input_index_list = range(len(arr_list))
        
        #The inputs can also be specified by ID
        elif input_ID_list is not None:
            input_index_list = [self.ID_dict[ID] for ID in input_ID_list]
        
        #If this is a new cycle:
        if new_cycle:
            #Increment the input counter.
            self.input_cycles += 1
    
            #Update the duty cycle
            self.duty_cycle_update()
        
            #Periodically increment the permanences for minicolumns with low duty cycles
            if (self.input_cycles % self.duty_cycle_period == 0) and (sp_learning):
                self.low_duty_cycle_inc()
            
            #Update the boost factors
            if boosting:
                self.update_boost_factors()
        
        #Get the minicolumn overlap scores based on all input SDRs
        overlap_scores = np.zeros(self.active_cols.shape)
        for j in input_index_list:
            overlap_scores += self.compute_overlap_par(arr_list[j], j)
            
        #Boost the scores, if necessary.
        if boosting:
            overlap_scores = overlap_scores*self.boost_factors
            
        #Determine the active minicolumns based on the overlap scores
        self.get_active_columns(overlap_scores);
        
        #Update the permanences for each minicolumn
        if sp_learning:
            for j in input_index_list:
                self.permanence_update(arr_list[j],j)
                
        #Return the active column set
        return self.active_cols
    
    def save_SP(self, path = '', name = 'Spatial_Pooler', string = ''):
        #Saves a copy of the SP to a file.
        filename = path + name + string + ".txt"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
