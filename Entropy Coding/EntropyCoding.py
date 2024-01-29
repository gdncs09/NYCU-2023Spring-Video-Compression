import cv2
import numpy as np

def zigzag_order(N, input, service):
    if service == 'encode':
        output = np.zeros((N*N))
    elif service == 'decode':
        output = np.zeros((N, N))
    x, y = 0, 0
    for i in range(N*N):
        if service == 'encode':
            output[i] = input[x, y]
        elif service == 'decode':
            output[x, y] = input[i]
        if (x + y) % 2 == 0:
            if y == N-1:
                x += 1
            elif x == 0:
                y += 1
            else:
                x -= 1
                y += 1
        else:
            if x == N-1:
                y += 1
            elif y == 0:
                x += 1
            else:
                x += 1
                y -= 1
    return output
    
def RLE_encoding(f_block, rle):
    count = 0
    for pixel in f_block:
        if pixel == 0:
            count += 1
        else:
            rle.append((int(count), int(pixel)))
            count = 0
    if count > 0:
        rle.append((int(count), 0))
    return rle

def RLE_decode(encoded, h, w, block_size):
    decoded_img = np.zeros((h, w), np.uint8)
    decoded_arr = []
    blocks = []
    sum = 0
    for value in encoded:
        count, pixel = value[0], value[1]
        decoded_arr.extend([0]*count)
        sum += (count)
        if pixel != 0:
            decoded_arr.extend([pixel])
            sum += 1 #count total value in decoded_arr
        if sum == block_size*block_size: 
            decoded_block = zigzag_order(block_size, decoded_arr, 'decode')
            decoded_arr = []
            sum = 0
            blocks.append(decoded_block)
    index = 0 #blocks index
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            decoded_quantized = np.multiply(blocks[index], quantization_matrix)
            decoded_img[i:i+block_size, j:j+block_size] = cv2.idct(decoded_quantized)
            index += 1
    return decoded_img
    
if __name__ == '__main__':
    img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    encoded_img = np.zeros_like(img)
    decoded_img = np.zeros_like(img)
    block_size = 8
    
    encoded = []
    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])
    #ENCODE
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
             block = np.float32(img[i:i+block_size, j:j+block_size]) 
             dct = cv2.dct(block)
             dct_quantized = np.divide(dct, quantization_matrix).astype(int)
             zz_block = zigzag_order(block_size, dct_quantized, 'encode')
             encoded_img[i:i+block_size, j:j+block_size] = np.reshape(zz_block, (block_size, block_size))
             encoded = RLE_encoding(zz_block, encoded)
    
    encoded = np.array(encoded)
    np.savez("np.npz", encoded)
    cv2.imshow('Encode DCT', np.uint8(encoded_img))
    #cv2.imwrite('encode.png', encoded_img)
    #DECODE
    decoded_img = RLE_decode(encoded, h, w, block_size)
    cv2.imshow('Decode', decoded_img)
    #cv2.imwrite('decode.png', decoded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()