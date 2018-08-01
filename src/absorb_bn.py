'''
Created on Feb 6, 2018

@author: guoweiyu
'''
import numpy as np 
import sys
import os.path as osp
from argparse import ArgumentParser
#sys.path.insert(0, '/home/guoweiyu/software/caffe/python')
caffe_root = '/home/guoweiyu/caffe_pixel_shuffle/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import google.protobuf as pb
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter
import cv2
import os

to_delete_empty = []

def load_and_fill_biases(src_model, src_weights, dst_model, dst_weights):  
    with open(src_model) as f:  
        model = caffe.proto.caffe_pb2.NetParameter()  
        pb.text_format.Merge(f.read(), model)  
  
    for i, layer in enumerate(model.layer):  
        if layer.type == 'Convolution' or layer.type == 'Deconvolution':  
            # Add bias layer if needed  
            print layer.convolution_param.bias_term ,layer.type
            if layer.convolution_param.bias_term == False:
                
                layer.convolution_param.bias_term = True  
                layer.convolution_param.bias_filler.type = 'constant'  
                layer.convolution_param.bias_filler.value = 0.0  
    #exit(0)
    with open(dst_model, 'w') as f:  
        f.write(pb.text_format.MessageToString(model))  
  
    caffe.set_mode_cpu()
    net_src = caffe.Net(src_model, src_weights, caffe.TEST)  
    net_dst = caffe.Net(dst_model, caffe.TEST)
    # copy the original weights to dst model
    for key in net_src.params.keys():  
        for i in range(len(net_src.params[key])):  
            net_dst.params[key][i].data[:] = net_src.params[key][i].data[:]  
  
    if dst_weights is not None:  
        # Store params  
        pass  
  
    return net_dst  
  
  
def merge_conv_and_bn(net, i_conv, i_bn, i_scale):  
    # This is based on Kyeheyon's work  
    assert(i_conv != None)  
    assert(i_bn != None)  
  
    def copy_double(data):  
        return np.array(data, copy=True, dtype=np.double)  
    
    key_conv = net._layer_names[i_conv]  
    key_bn = net._layer_names[i_bn]  
    key_scale = net._layer_names[i_scale] if i_scale else None
    
    # Copy  
    bn_mean = copy_double(net.params[key_bn][0].data)  
    bn_variance = copy_double(net.params[key_bn][1].data)  
    num_bn_samples = copy_double(net.params[key_bn][2].data) 
  
    # and Invalidate the BN layer  
    net.params[key_bn][0].data[:] = 0  
    net.params[key_bn][1].data[:] = 1  
    net.params[key_bn][2].data[:] = 1  
    if num_bn_samples[0] == 0:  
        num_bn_samples[0] = 1  
  
    if net.params.has_key(key_scale):  
        print 'Combine {:s} + {:s} + {:s}'.format(key_conv, key_bn, key_scale)  
        scale_weight = copy_double(net.params[key_scale][0].data)  
        scale_bias = copy_double(net.params[key_scale][1].data)  
        net.params[key_scale][0].data[:] = 1  
        net.params[key_scale][1].data[:] = 0  
    else:  
        print 'Combine {:s} + {:s}'.format(key_conv, key_bn)  
        scale_weight = 1  
        scale_bias = 0  
    
             
    weight = copy_double(net.params[key_conv][0].data)  
    bias = copy_double(net.params[key_conv][1].data)  
    alpha = scale_weight / np.sqrt(bn_variance / num_bn_samples[0] + np.finfo(np.double).eps)  
    net.params[key_conv][1].data[:] = bias * alpha + (scale_bias - (bn_mean / num_bn_samples[0]) * alpha)  
    
    if net.layers[i_conv].type == 'Convolution':
        for i in range(len(alpha)): 
            net.params[key_conv][0].data[i] = weight[i] * alpha[i]
    #the shape of deconvolution weight is (input-chs, output-chs, kernel_size)
    elif net.layers[i_conv].type == 'Deconvolution':
        step_size = weight.shape[0] / bias.shape[0]
        print step_size,bias.shape,weight.shape
       
        for i in range(len(alpha)):
            for j in range(step_size):
                net.params[key_conv][0].data[i * step_size + j] = weight[i * step_size + j] * alpha[i]
        
    
    
def merge_batchnorms_in_net(net):  
    # for each BN  
    for i, layer in enumerate(net.layers):  
        if layer.type != 'BatchNorm':  
            continue  
  
        l_name = net._layer_names[i]  
  
        l_bottom = net.bottom_names[l_name]  
        assert(len(l_bottom) == 1)  
        l_bottom = l_bottom[0]  
        l_top = net.top_names[l_name]  
        assert(len(l_top) == 1)  
        l_top = l_top[0]  
  
        can_be_absorbed = True  
  
        # Search all (bottom) layers  
        for j in xrange(i - 1, -1, -1):  
            tops_of_j = net.top_names[net._layer_names[j]]  
            if l_bottom in tops_of_j:  
                if net.layers[j].type not in ['Convolution', 'InnerProduct', 'Deconvolution']:  
                    can_be_absorbed = False  
                else:  
                    # There must be only one layer  
                    conv_ind = j  
                    break  
  
        if not can_be_absorbed:  
            continue  
  
        # find the following Scale  
        scale_ind = None  
        for j in xrange(i + 1, len(net.layers)):  
            bottoms_of_j = net.bottom_names[net._layer_names[j]]  
            if l_top in bottoms_of_j:  
                if scale_ind:  
                    # Followed by two or more layers  
                    scale_ind = None  
                    break  
  
                if net.layers[j].type in ['Scale']:  
                    scale_ind = j  
  
                    top_of_j = net.top_names[net._layer_names[j]][0]  
                    if top_of_j == bottoms_of_j[0]:  
                        # On-the-fly => Can be merged  
                        break  
                else:  
                    # Followed by a layer which is not 'Scale'  
                    scale_ind = None  
                    break  
  
        merge_conv_and_bn(net, conv_ind, i, scale_ind)  
    return net  
  

def post_process_prototxt(net, model):
    del(to_delete_empty[:])
    
    for i, layer in enumerate(model.layer):  
        if layer.type != 'BatchNorm':
            continue
        
        l_name = layer.name
        l_bottom = layer.bottom
        assert(len(l_bottom) == 1)  
        l_bottom = l_bottom[0]  
        l_top = layer.top 
        assert(len(l_top) == 1)  
        l_top = l_top[0]  
        

        zero_mean = np.all(net.params[l_name][0].data == 0)  
        one_var = np.all(net.params[l_name][1].data == 1)  
        # length_is_1 = (net.params['conv1_1/bn'][2].data == 1) or (net.params[layer.name][2].data == 0)  
        length_is_1 = (net.params[l_name][2].data == 1)  
       
        if zero_mean and one_var and length_is_1:  
            print 'Delete layer: {}'.format(l_name)  
            # to_delete_empty.append(layer)
            to_delete_empty.append(layer)
            
            '''
            # find the index of current bn_layer in net
            for i , bn_layer in enumerate(net.layers): 
                if bn_layer.type != 'BatchNorm':
                    continue
                if l_name == net._layer_names[i]:
                    break  
            '''
            active_ind = None
            # find the following RuLU layer
            for j in xrange(i + 1, len(model.layer)):
                bottoms_of_j = model.layer[j].bottom  # net.bottom_names[net._layer_names[j]]
                
                if l_top in bottoms_of_j:  
                    if active_ind:  
                        # Followed by two or more layers  
                        active_ind = None  
                        break  
                    
                    if model.layer[j].type in ['ReLU','Sigmoid','Deconvolution']:  
                        active_ind = j  
                        # top_of_j = model.layer[j].top#net.top_names[net._layer_names[j]][0]  
                        # if top_of_j[0] == bottoms_of_j[0]:  
                        if len(bottoms_of_j) == 1:  
                            # On-the-fly => Can be merged  
                            break     
                    else:  
                        # Followed by a layer which is not 'ReLU'  
                        active_ind = None  
                    
            if active_ind is not None:
                act_layer = model.layer[active_ind]  # net.layers[active_ind]
                act_layer.bottom[0] = l_bottom
                # act_layer.top[0] = l_bottom
    print len(to_delete_empty)
 

def process_model(net, src_model, dst_model, func_loop, func_finally):  
    with open(src_model) as f:  
        model = caffe.proto.caffe_pb2.NetParameter()  
        pb.text_format.Merge(f.read(), model)  
    
    for i, layer in enumerate(model.layer):  
        map(lambda x: x(layer, net, model, i), func_loop)  
   
    map(lambda x: x(net, model), func_finally) 
    
    post_process_prototxt(net, model)
    map(lambda x: x(net, model), func_finally) 
    
    with open(dst_model, 'w') as f:  
        f.write(pb.text_format.MessageToString(model))  
  
  
# Functions to remove (redundant) BN and Scale layers   
def pick_empty_layers(layer, net, model, i):  
    if layer.type not in ['BatchNorm', 'Scale']:  
        return  

    bottom = layer.bottom[0]  
    top = layer.top[0]

    if (bottom != top):  
        # Not supperted yet  
        return  

    if layer.type == 'BatchNorm':
        zero_mean = np.all(net.params[layer.name][0].data == 0)  
        one_var = np.all(net.params[layer.name][1].data == 1)  
        # length_is_1 = (net.params['conv1_1/bn'][2].data == 1) or (net.params[layer.name][2].data == 0)  
        length_is_1 = (net.params[layer.name][2].data == 1)  
       
        if zero_mean and one_var and length_is_1:  
            print 'Delete layer: {}'.format(layer.name)  
            to_delete_empty.append(layer)
            
    if layer.type == 'Scale':  
        no_scaling = np.all(net.params[layer.name][0].data == 1)  
        zero_bias = np.all(net.params[layer.name][1].data == 0)  
  
        if no_scaling and zero_bias:  
            print 'Delete layer: {}'.format(layer.name)  
            to_delete_empty.append(layer)  

 
def remove_empty_layers(net, model):  
    map(model.layer.remove, to_delete_empty)  
  
  
# A function to add 'engine: CAFFE' param into 1x1 convolutions  
def set_engine_caffe(layer, net, model, i):  
    if layer.type == 'Convolution':  
        if layer.convolution_param.kernel_size == 1 \
        or (layer.convolution_param.kernel_h == layer.convolution_param.kernel_w == 1):  
            layer.convolution_param.engine = dict(layer.convolution_param.Engine.items())['CAFFE']  
  
  
def main(args):  
    # Set default output file names  
    if args.output_model is None:  
        file_name = osp.splitext(args.model)[0]  
        args.output_model = file_name + '_inference.prototxt'  
    
    if args.output_weights is None:  
        file_name = osp.splitext(args.weights)[0]  
        args.output_weights = file_name + '_inference.caffemodel'  
  
    net = load_and_fill_biases(args.model, args.weights, args.model + '.temp.pt', None)  
    net = merge_batchnorms_in_net(net)  
    process_model(net, args.model + '.temp.pt', args.output_model,
                  [pick_empty_layers, set_engine_caffe],
                  [remove_empty_layers])  
    # Store params  
    net.save(args.output_weights)  


def test4seg(args):
    pro_threshold = 0.8
    img_path = '/home/guoweiyu/workspace/PeopleSeg/debug/cs.jpg'
    input_width = 100
    input_height = 150
    mean = np.array([104.008, 116.669, 122.675])
    
    im = cv2.imread(img_path)
    h, w = im.shape[:2]
    im = cv2.resize(im, (input_width, input_height))
    im = im - mean
    im = im.transpose((2, 0, 1))
    net_file = '/home/guoweiyu/workspace/PeopleSeg/debug/psg_mob_32.prototxt' 
    caffe_model = '/home/guoweiyu/workspace/PeopleSeg/debug/psg_mob_32.caffemodel'
    
    net_file = '/home/guoweiyu/workspace/PeopleSeg/debug/psg_half_32bn_inference.prototxt' 
    caffe_model = '/home/guoweiyu/workspace/PeopleSeg/debug/psg_half_32bn_inference.caffemodel'
    
    if args.output_model is None:  
        file_name = osp.splitext(args.model)[0]  
        net_file = file_name + '_inference.prototxt'  
    if args.output_weights is None:  
        file_name = osp.splitext(args.weights)[0]  
        caffe_model = file_name + '_inference.caffemodel'
    
    print net_file, caffe_model
    net = caffe.Net(net_file, caffe_model, caffe.TEST)
    net.blobs['data'].data[...] = im
    out = net.forward()
    
    #pred_img = out['ConvNdBackward72'].reshape((input_height, input_width))
    pred_img = out['BatchNormBackward42'].reshape((input_height, input_width))
    pred_img = cv2.resize(pred_img, (w , h))
    temp1 = np.zeros((h , w , 3), dtype=np.uint8)
    
    temp1[:, :, 0][pred_img > pro_threshold] = 255
    temp1[:, :, 1][pred_img > pro_threshold] = 255
    temp1[:, :, 2][pred_img > pro_threshold] = 255
    cv2.imwrite(os.path.join('../debug', 'debug.jpg'), temp1)   
    

if __name__ == '__main__':  
    parser = ArgumentParser(
            description="Generate Batch Normalized model for inference")  
    parser.add_argument('--model', help="The net definition prototxt")  
    parser.add_argument('--weights', help="The weights caffemodel")  
    parser.add_argument('--output_model')  
    parser.add_argument('--output_weights')  
    args = parser.parse_args()  
    main(args)
    
    #test4seg(args)
