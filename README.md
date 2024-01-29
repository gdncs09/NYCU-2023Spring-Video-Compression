# 2023Spring-Video-Compression
## Homework #1 – Color Transform
Please represent “lena.png” in terms of RGB, YUV, and YCbCr.
RGB -> YUV: 

\begin{array}{rll}
Y &= 0.299 * R + 0.587 * G + 0.114 * B \\
U &= -0.169 * R - 0.331 * G + 0.5 * B + 128 \\
V &= 0.5 * R - 0.419 * G - 0.081 * B + 128
\end{array}
RGB -> YCbCr: in the slides
In any programming language you are comfortable with (C/C++/Python/MATLAB).
Output 8 grayscale images representing R, G, B, Y, U, V, Cb, Cr, respectively.
Do not use any ready-made functions to transform the color.
You are allowed to use image reading/writing APIs.