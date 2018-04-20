import cv2
import numpy as np

class ImageDataGenerator_landmarks(object):
    def __init__(self,
                 datagen,
                 preprocessing_function= lambda x,y: (x,y),
                 loc_xRE=None, 
                 loc_xLE=None,
                 flip_indicies=None,
                 target_shape=None,
                 ignore_horizontal_flip=True):
        '''
        datagen : Keras's ImageDataGenerator
        preprocessing_function : The function that will be implied on each input. 
                                 The function will run after the image is resized and augmented. 
                                 The function should take one argument: one image (Numpy tensor with rank 3), 
                                 and should output a Numpy 
        ignore_horizontal_flip : if False, whether the horizontal flip happend is checked 
                                 using <loc_xRE> and <loc_xLE>
                                 if the flipping happens, 
                                 each pair of the <flip_indicies> are flipped.
                                 if True, then <flip_indicies>, 
                                 <loc_xRE> and <loc_xLE> do not need to be specified.
                                
        target_shape            : If target_shape is not None,
                                  A translated image is resized to target_shape. 
                                  Why? Translation with original resolution and then down size resolution 
                                       gives wider range of modified images than translating the down sized images.
    
        For example,
        
        Suppose the landmarks are 
        
        - right eye (RE) 
        - left eye (LE)
        - mouth (M)
        - right mouth edge (RM)
        - left mouth edge (LM)
        
        then there are 5 x 2 coordinates to predict:
        
        xRE, yRE, xLE, yLE, xN, yN, xRM, yRM, xLM, yLM
        
        When the horizontal flip happens, RE becomes LE and RM becomes LM.
        So we need to change the target variables accordingly.
        
        If the horizontal flip happenes  xRE > xLE
        so loc_xRE = 0 , loc_yRE = 2
        
        In this case, our filp indicies are:
        
        self.flip_indicies =  ((0,2), # xRE <-> xLE
                               (1,3), # yRE <-> yLE
                               (6,8), # xRM <-> xLM
                               (7,9)) # yRM <-> yLM

        '''
        self.datagen = datagen
        self.ignore_horizontal_flip = ignore_horizontal_flip
        self.target_shape = target_shape
        # check if x-cord of landmark1 is less than x-cord of landmark2
        self.loc_xRE, self.loc_xLE = loc_xRE, loc_xLE
        
        self.flip_indicies = flip_indicies
        ## the chanel that records the mask
        self.loc_mask = 3

        self.preprocessing_function = preprocessing_function
        
    def flow(self,imgs,batch_size=20):
        '''
        imgs: the numpy image array : (batch, height, width, image channels + 1)
              the channel (self.loc_mask)th channel must contain mask
        '''
        
        generator = self.datagen.flow(imgs,batch_size=batch_size)
        while 1:
            ## 
            N = 0
            x_bs, y_bs = [], [] 
            while N < batch_size:
                yimgs = generator.next() 
                ## yimgs.shape = (bsize,width,height,channels + 1)
                ## where bsize = np.min(batch_size,x.shape[0])
                x_batch ,y_batch = self._keep_only_valid_image(yimgs)
                if len(x_batch) == 0:
                    continue
                x_batch ,y_batch = self.preprocessing_function(x_batch,y_batch)
                x_bs.append(x_batch)
                y_bs.append(y_batch)
                N += x_batch.shape[0]
            x_batch , y_batch = np.vstack(x_bs), np.vstack(y_bs)
            yield ([x_batch, y_batch])


    def _keep_only_valid_image(self,yimg):
        '''
        Transform the mask to (x,y)-coordiantes.
        Depending on the translation, landmark may "dissapeear".
        For example, if the image is escessively zoomed in, 
        the mask may lose the index of landmark.
        Such image translation is discarded.
        
        x_train and y_train could be an empty array 
        if landmarks of all the translated images are lost i.e.
        np.array([])
        '''
        x_train, y_train = [], []
        for irow in range(yimg.shape[0]):
            x     = yimg[irow,:,:,:self.loc_mask]
            ymask = yimg[irow,:,:,self.loc_mask]
            y     = self._findindex_from_mask(ymask)
            # if some landmarks dissapears, do not use the translated image 
            if y is None:
                continue
            x, y  = self._resize_image(x, y)    
            x_train.append(x)
            y_train.append(y)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return(x_train,y_train)
    
    def _resize_image(self,x,y):
        '''
        this function is useful for down scaling the resolution
        '''
        if self.target_shape is not None:
            shape_orig = x.shape
            x = cv2.resize(x,self.target_shape[:2])
            
            y = self.adjust_xy(y,
                               shape_orig,
                               self.target_shape)
        return x,y
    def adjust_xy(self,y,shape_orig,shape_new):
        '''
        y : [x1,y1,x2,y2,...]
        '''
        y[0::2] = y[0::2]*shape_new[1]/float(shape_orig[1])
        y[1::2] = y[1::2]*shape_new[0]/float(shape_orig[0])
        return y

    def _findindex_from_mask(self,ymask):
        '''
        ymask : a mask of shape (height, width, 1)
        '''
        
        ys = []
        for i in range(self.Nlandmarks):
            ix, iy = np.where(ymask==i)
            if len(ix) == 0:
                return(None)
            ys.extend([np.mean(iy),
                       np.mean(ix)])
        ys = np.array(ys)
        ys = self._adjustLR_horizontal_flip(ys)
        return(ys)

    def _adjustLR_horizontal_flip(self,ys):
        '''
        if a horizontal flip happens, 
        right eye becomes left eye and 
        right mouth edge becomes left mouth edge
        So we need to flip the target cordinates accordingly
        '''
        if self.ignore_horizontal_flip:
            return(ys)
        
        if ys[self.loc_xRE] > ys[self.loc_xLE]: ## True if flip happens
            # x-cord of RE is less than x-coord of left eye
            # horizontal flip happened!
            for a, b in self.flip_indicies:
                ys[a],ys[b] = (ys[b],ys[a])
        return(ys)

    def get_ymask(self,img, xys):
        '''
        img : (height, width, channels) array of image
        xys : A list containint tuple of (x,y) coordinates of landmark. For example:

        xys = [(x0,y0),
               (x1,y1),
               (x2,y2),
               (x3,y3),
               (x4,y4),
               ...] 
        output:
        
        mask : A numpy array of size (height, width, channels). 
               All locations without the landmarks are recorded -1 
               A coordinate with (x0, y0) is recorded as 0
               A coordinate with (x1, y1) is recorded as 1
               ...
        
        '''
        yimg = np.zeros((img.shape[0],img.shape[1],1))
        yimg[:] = -1
        for iland, (ix,iy) in enumerate(xys):
            yimg[iy,ix] = iland
        
        self.Nlandmarks = len(xys)
        self.loc_mask   = img.shape[2] 
        return(np.dstack([img,yimg]))