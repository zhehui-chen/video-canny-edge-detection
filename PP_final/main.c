#include <stdio.h>
#include <stdlib.h>

typedef unsigned char uchar;

#include <assert.h>
#include <stdbool.h>
#include <math.h>

#define MAX_BRIGHTNESS 255
#define CANNY_LOWER 45
#define CANNY_UPPER 50
#define CANNY_SIGMA 1.0

/* Use short int instead unsigned char so that we can store negative values. */
typedef float pixel_t;

void storeImage(float *imageOut, const char *filename, int rows, int cols,
                const char *refFilename)
{

   FILE *ifp, *ofp;
   unsigned char tmp;
   int offset;
   unsigned char *buffer;
   int i, j;

   int bytes;

   int height, width;

   ifp = fopen(refFilename, "rb");
   if (ifp == NULL)
   {
      perror(filename);
      exit(-1);
   }

   fseek(ifp, 10, SEEK_SET);
   fread(&offset, 4, 1, ifp);

   fseek(ifp, 18, SEEK_SET);
   fread(&width, 4, 1, ifp);
   fread(&height, 4, 1, ifp);

   fseek(ifp, 0, SEEK_SET);

   buffer = (unsigned char *)malloc(offset);
   if (buffer == NULL)
   {
      perror("malloc");
      exit(-1);
   }

   fread(buffer, 1, offset, ifp);

   printf("Writing output image to %s\n", filename);
   ofp = fopen(filename, "wb");
   if (ofp == NULL)
   {
      perror("opening output file");
      exit(-1);
   }
   bytes = fwrite(buffer, 1, offset, ofp);
   if (bytes != offset)
   {
      printf("error writing header!\n");
      exit(-1);
   }

   // NOTE bmp formats store data in reverse raster order (see comment in
   // readImage function), so we need to flip it upside down here.
   int mod = width % 4;
   if (mod != 0)
   {
      mod = 4 - mod;
   }
   //   printf("mod = %d\n", mod);
   for (i = height - 1; i >= 0; i--)
   {
      for (j = 0; j < width; j++)
      {
         tmp = (unsigned char)imageOut[i * cols + j];
         fwrite(&tmp, sizeof(char), 1, ofp);
      }
      // In bmp format, rows must be a multiple of 4-bytes.
      // So if we're not at a multiple of 4, add junk padding.
      for (j = 0; j < mod; j++)
      {
         fwrite(&tmp, sizeof(char), 1, ofp);
      }
   }

   fclose(ofp);
   fclose(ifp);

   free(buffer);
}

/*
 * If normalize is true, then map pixels to range 0 -> MAX_BRIGHTNESS.
 */
static void
convolution(const pixel_t *in,
            pixel_t       *out,
            const float   *kernel,
            const int      nx,
            const int      ny,
            const int      kn,
            const bool     normalize)
{
  const int khalf = kn / 2;
  float min = 0.5;
  float max = 254.5;
  float pixel = 0.0;
  size_t c = 0;
  int m, n, i, j;

  assert(kn % 2 == 1);
  assert(nx > kn && ny > kn);

  for (m = khalf; m < nx - khalf; m++) {
    for (n = khalf; n < ny - khalf; n++) {
      pixel = c = 0;

      for (j = -khalf; j <= khalf; j++)
        for (i = -khalf; i <= khalf; i++)
          pixel += in[(n - j) * nx + m - i] * kernel[c++];

      if (normalize == true)
        pixel = MAX_BRIGHTNESS * (pixel - min) / (max - min);

      out[n * nx + m] = (pixel_t) pixel;
    }
  }
}

/*
 * gaussianFilter: http://www.songho.ca/dsp/cannyedge/cannyedge.html
 * Determine the size of kernel (odd #)
 * 0.0 <= sigma < 0.5 : 3
 * 0.5 <= sigma < 1.0 : 5
 * 1.0 <= sigma < 1.5 : 7
 * 1.5 <= sigma < 2.0 : 9
 * 2.0 <= sigma < 2.5 : 11
 * 2.5 <= sigma < 3.0 : 13 ...
 * kernel size = 2 * int(2 * sigma) + 3;
 */
static void
gaussian_filter(const pixel_t *in,
                pixel_t       *out,
                const int      nx,
                const int      ny,
                const float    sigma)
{
  const int n = 2 * (int) (2 * sigma) + 3;
  const float mean = (float) floor(n / 2.0);
  float kernel[n * n];
  int i, j;
  size_t c = 0;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++)
      kernel[c++] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) + pow((j - mean) / sigma, 2.0))) / (2 * M_PI * sigma * sigma);
  }

  convolution(in, out, kernel, nx, ny, n, true);
}

/*
 * Links:
 * http://en.wikipedia.org/wiki/Canny_edge_detector
 * http://www.tomgibara.com/computer-vision/CannyEdgeDetector.java
 * http://fourier.eng.hmc.edu/e161/lectures/canny/node1.html
 * http://www.songho.ca/dsp/cannyedge/cannyedge.html
 *
 * Note: T1 and T2 are lower and upper thresholds.
 */

static float *
canny_edge_detection(const float *in,
                     const int      width,
                     const int      height,
                     const int      t1,
                     const int      t2,
                     const float    sigma)
{
  int i, j, k, nedges;
  int *edges;
  size_t t = 1;
  float *retval;

  const float Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const float Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

  pixel_t *G = calloc(width * height * sizeof(pixel_t), 1);

  pixel_t *after_Gx = calloc(width * height * sizeof(pixel_t), 1);

  pixel_t *after_Gy = calloc(width * height * sizeof(pixel_t), 1);

  pixel_t *nms = calloc(width * height * sizeof(pixel_t), 1);

  pixel_t *out = malloc(width * height * sizeof(pixel_t));

  pixel_t *pixels = malloc(width * height * sizeof(pixel_t));

  /* Convert to pixel_t. */
  for (i = 0; i < width * height; i++) {
    pixels[i] = (pixel_t)in[i];
  }

  gaussian_filter(pixels, out, width, height, sigma);

  convolution(out, after_Gx, Gx, width, height, 3, false);

  convolution(out, after_Gy, Gy, width, height, 3, false);

  for (i = 1; i < width - 1; i++) {
    for (j = 1; j < height - 1; j++) {
      const int c = i + width * j;
      G[c] = (pixel_t)hypot(after_Gx[c], after_Gy[c]);
    }
  }

  /* Non-maximum suppression, straightforward implementation. */
  for (i = 1; i < width - 1; i++) {
    for (j = 1; j < height - 1; j++) {
      const int c = i + width * j;
      const int nn = c - width;
      const int ss = c + width;
      const int ww = c + 1;
      const int ee = c - 1;
      const int nw = nn + 1;
      const int ne = nn - 1;
      const int sw = ss + 1;
      const int se = ss - 1;
      const float dir = (float) (fmod(atan2(after_Gy[c], after_Gx[c]) + M_PI, M_PI) / M_PI) * 8;

      if (((dir <= 1 || dir > 7) && G[c] > G[ee] && G[c] > G[ww]) || // 0 deg
          ((dir > 1 && dir <= 3) && G[c] > G[nw] && G[c] > G[se]) || // 45 deg
          ((dir > 3 && dir <= 5) && G[c] > G[nn] && G[c] > G[ss]) || // 90 deg
          ((dir > 5 && dir <= 7) && G[c] > G[ne] && G[c] > G[sw]))   // 135 deg
        nms[c] = G[c];
      else
        nms[c] = 0;
    }
  }

  /* Reuse the array used as a stack, width * height / 2 elements should be enough. */
  edges = (int *) after_Gy;
  memset(out, 0, sizeof(pixel_t) * width * height);
  memset(edges, 0, sizeof(pixel_t) * width * height);

  /* Tracing edges with hysteresis. Non-recursive implementation. */
  for (j = 1; j < height - 1; j++) {
    for (i = 1; i < width - 1; i++) {
      /* Trace edges. */
      if (nms[t] >= t2 && out[t] == 0) {
        out[t] = MAX_BRIGHTNESS;
        nedges = 1;
        edges[0] = t;

        do {
          nedges--;
          const int e = edges[nedges];

          int nbs[8]; // neighbours
          nbs[0] = e - width;     // nn
          nbs[1] = e + width;     // ss
          nbs[2] = e + 1;      // ww
          nbs[3] = e - 1;      // ee
          nbs[4] = nbs[0] + 1; // nw
          nbs[5] = nbs[0] - 1; // ne
          nbs[6] = nbs[1] + 1; // sw
          nbs[7] = nbs[1] - 1; // se

          for (k = 0; k < 8; k++) {
            if (nms[nbs[k]] >= t1 && out[nbs[k]] == 0) {
              out[nbs[k]] = MAX_BRIGHTNESS;
              edges[nedges] = nbs[k];
              nedges++;
            }
          }
        } while (nedges > 0);
      }
      t++;
    }
  }

  retval = malloc(width * height * sizeof(float));

  /* Convert back to float */
  for (i = 0; i < width * height; i++) {
    retval[i] = (float)out[i];
  }

  free(after_Gx);
  free(after_Gy);
  free(G);
  free(nms);
  free(pixels);
  free(out);

  return retval;
}

float *readImage(const char *filename, int *widthOut, int *heightOut)
{

   uchar *imageData;

   int height, width;
   uchar tmp;
   int offset;
   int i, j;

   printf("Reading input image from %s\n", filename);
   FILE *fp = fopen(filename, "rb");
   if (fp == NULL)
   {
      perror(filename);
      exit(-1);
   }

   fseek(fp, 10, SEEK_SET);
   fread(&offset, 4, 1, fp);

   fseek(fp, 18, SEEK_SET);
   fread(&width, 4, 1, fp);
   fread(&height, 4, 1, fp);

   printf("width = %d\n", width);
   printf("height = %d\n", height);

   *widthOut = width;
   *heightOut = height;

   imageData = (uchar *)malloc(width * height);
   if (imageData == NULL)
   {
      perror("malloc");
      exit(-1);
   }

   fseek(fp, offset, SEEK_SET);
   fflush(NULL);

   int mod = width % 4;
   if (mod != 0)
   {
      mod = 4 - mod;
   }

   // NOTE bitmaps are stored in upside-down raster order.  So we begin
   // reading from the bottom left pixel, then going from left-to-right,
   // read from the bottom to the top of the image.  For image analysis,
   // we want the image to be right-side up, so we'll modify it here.

   // First we read the image in upside-down

   // Read in the actual image
   for (i = 0; i < height; i++)
   {

      // add actual data to the image
      for (j = 0; j < width; j++)
      {
         fread(&tmp, sizeof(char), 1, fp);
         imageData[i * width + j] = tmp;
      }
      // For the bmp format, each row has to be a multiple of 4,
      // so I need to read in the junk data and throw it away
      for (j = 0; j < mod; j++)
      {
         fread(&tmp, sizeof(char), 1, fp);
      }
   }

   // Then we flip it over
   int flipRow;
   for (i = 0; i < height / 2; i++)
   {
      flipRow = height - (i + 1);
      for (j = 0; j < width; j++)
      {
         tmp = imageData[i * width + j];
         imageData[i * width + j] = imageData[flipRow * width + j];
         imageData[flipRow * width + j] = tmp;
      }
   }

   fclose(fp);

   // Input image on the host
   float *floatImage = NULL;
   floatImage = (float *)malloc(sizeof(float) * width * height);
   if (floatImage == NULL)
   {
      perror("malloc");
      exit(-1);
   }

   // Convert the BMP image to float (not required)
   for (i = 0; i < height; i++)
   {
      for (j = 0; j < width; j++)
      {
         floatImage[i * width + j] = (float)imageData[i * width + j];
      }
   }

   free(imageData);
   return floatImage;
}


int main(int argc, char **argv){
     const char *inputFile = "input.bmp";
     const char *outputFile = "output.bmp";
     const char *output2File = "output2.bmp";
     int imageWidth, imageHeight;
     float *inputImage = readImage(inputFile, &imageWidth, &imageHeight);
     float *output = (float*)malloc(imageWidth * imageHeight * sizeof(float));

     output = canny_edge_detection(inputImage, imageWidth, imageHeight, CANNY_LOWER, CANNY_UPPER, CANNY_SIGMA);
     storeImage(output, outputFile, imageHeight, imageWidth, inputFile);

     output = canny_edge_detection(inputImage, imageWidth, imageHeight, 10, CANNY_UPPER, 0.1);
     storeImage(output, output2File, imageHeight, imageWidth, inputFile);

     free(output);
}
