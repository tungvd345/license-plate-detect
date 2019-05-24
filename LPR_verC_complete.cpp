#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>


#define SOD_IMG_GRAYSCALE 1 /* Load an image in the grayscale colorpsace only (single channel). */
#define LOW_THRESHOLD_PERCENTAGE 0.8 /* percentage of the high threshold value that the low threshold shall be set at */
#define HIGH_THRESHOLD_PERCENTAGE 0.10 /* percentage of pixels that meet the high threshold - for example 0.15 will ensure that at least 15% of edge pixels are considered to meet the high threshold */
#define SOD_OK           0 /* Everything went well */
#define SOD_UNSUPPORTED -1 /* Unsupported Pixel format */
#define SOD_OUTOFMEM    -4 /* Out-of-Memory */
#define SOD_ABORT	    -5 /* User callback request an operation abort */
#define SOD_IOERR       -6 /* IO error */
#define SOD_LIMIT       -7 /* Limit reached */
typedef struct sod_img sod_img;
struct sod_img {
	int h;   /* Image/frame height */
	int w;   /* Image/frame width */
	int c;   /* Image depth/Total number of color channels e.g. 1 for grayscale images, 3 RGB, etc. */
	float *data; /* Blob */
};

typedef struct sod_box sod_box;
struct sod_box {
	int x;  /* The x-coordinate of the upper-left corner of the rectangle */
	int y;  /* The y-coordinate of the upper-left corner of the rectangle */
	int w;  /* Rectangle width */
	int h;  /* Rectangle height */
	float score;       /* Confidence threshold. */
	const char *zName; /* Detected object name. I.e. person, face, dog, car, plane, cat, bicycle, etc. */
	void *pUserData;   /* External pointer used by some modules such as the face landmarks, NSFW classifier, pose estimator, etc. */
};

typedef struct sod_label_coord sod_label_coord;
struct sod_label_coord
{
	int xmin;
	int xmax;
	int ymin;
	int ymax;
	sod_label_coord *pNext; /* Next recorded label on the list */
};

sod_img sod_make_empty_image(int w, int h, int c)
{
	sod_img out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}
sod_img sod_make_image(int w, int h, int c)
{
	sod_img out = sod_make_empty_image(w, h, c);
	out.data = (float*)calloc(h*w*c, sizeof(float));
	return out;
}



sod_img sod_copy_image(sod_img m)
{
	sod_img copy = m;
	copy.data = (float*)calloc(m.h*m.w*m.c, sizeof(float));
	if (copy.data && m.data) {
		memcpy(copy.data, m.data, m.h*m.w*m.c * sizeof(float));
	}
	return copy;
}

static inline float get_pixel(sod_img m, int x, int y, int c)
{
	return (m.data ? m.data[c*m.h*m.w + y * m.w + x] : 0.0f);
}

sod_img sod_grayscale_image(sod_img im)
{
	if (im.c != 1) {
		int i, j, k;
		sod_img gray = sod_make_image(im.w, im.h, 1);
		if (gray.data) {
			float scale[] = { 0.587, 0.299, 0.114 };
			for (k = 0; k < im.c; ++k) {
				for (j = 0; j < im.h; ++j) {
					for (i = 0; i < im.w; ++i) {
						gray.data[i + im.w*j] += scale[k] * get_pixel(im, i, j, k);
					}
				}
			}
		}
		return gray;
	}
	return sod_copy_image(im); /* Already grayscaled */
}

sod_img sod_threshold_image(sod_img im, float thresh)
{
	sod_img t = sod_make_image(im.w, im.h, im.c);
	if (t.data) {
		int i;
		for (i = 0; i < im.w*im.h*im.c; ++i) {
			t.data[i] = im.data[i] > thresh ? 1.00000 : 0.00000;
		}
	}
	return t;
}

sod_img sod_gaussian_noise_reduce(sod_img grayscale)
{
	int w, h, x, y, max_x, max_y;
	sod_img img_out;
	if (!grayscale.data || grayscale.c != SOD_IMG_GRAYSCALE) {
		return sod_make_empty_image(0, 0, 0);
	}
	w = grayscale.w;
	h = grayscale.h;
	img_out = sod_make_image(w, h, 1);
	if (img_out.data) {
		max_x = w - 2;
		max_y = w * (h - 2);
		for (y = w * 2; y < max_y; y += w) {
			for (x = 2; x < max_x; x++) {
				img_out.data[x + y] = (2 * grayscale.data[x + y - 2 - w - w] +
					4 * grayscale.data[x + y - 1 - w - w] +
					5 * grayscale.data[x + y - w - w] +
					4 * grayscale.data[x + y + 1 - w - w] +
					2 * grayscale.data[x + y + 2 - w - w] +
					4 * grayscale.data[x + y - 2 - w] +
					9 * grayscale.data[x + y - 1 - w] +
					12 * grayscale.data[x + y - w] +
					9 * grayscale.data[x + y + 1 - w] +
					4 * grayscale.data[x + y + 2 - w] +
					5 * grayscale.data[x + y - 2] +
					12 * grayscale.data[x + y - 1] +
					15 * grayscale.data[x + y] +
					12 * grayscale.data[x + y + 1] +
					5 * grayscale.data[x + y + 2] +
					4 * grayscale.data[x + y - 2 + w] +
					9 * grayscale.data[x + y - 1 + w] +
					12 * grayscale.data[x + y + w] +
					9 * grayscale.data[x + y + 1 + w] +
					4 * grayscale.data[x + y + 2 + w] +
					2 * grayscale.data[x + y - 2 + w + w] +
					4 * grayscale.data[x + y - 1 + w + w] +
					5 * grayscale.data[x + y + w + w] +
					4 * grayscale.data[x + y + 1 + w + w] +
					2 * grayscale.data[x + y + 2 + w + w]) / 159;
			}
		}
	}
	return img_out;
}
static void canny_calc_gradient_sobel(sod_img * img_in, int *g, int *dir)
{
	int w, h, x, y, max_x, max_y, g_x, g_y;
	float g_div;
	w = img_in->w;
	h = img_in->h;
	max_x = w - 3;
	max_y = w * (h - 3);
	for (y = w * 3; y < max_y; y += w) {
		for (x = 3; x < max_x; x++) {
			g_x = (int)(255 * (2 * img_in->data[x + y + 1]
				+ img_in->data[x + y - w + 1]
				+ img_in->data[x + y + w + 1]
				- 2 * img_in->data[x + y - 1]
				- img_in->data[x + y - w - 1]
				- img_in->data[x + y + w - 1]));
			g_y = (int)(255 * (2 * img_in->data[x + y - w]
				+ img_in->data[x + y - w + 1]
				+ img_in->data[x + y - w - 1]
				- 2 * img_in->data[x + y + w]
				- img_in->data[x + y + w + 1]
				- img_in->data[x + y + w - 1]));

			g[x + y] = sqrt(g_x * g_x + g_y * g_y);

			if (g_x == 0) {
				dir[x + y] = 2;
			}
			else {
				g_div = g_y / (float)g_x;
				if (g_div < 0) {
					if (g_div < -2.41421356237) {
						dir[x + y] = 0;
					}
					else {
						if (g_div < -0.414213562373) {
							dir[x + y] = 1;
						}
						else {
							dir[x + y] = 2;
						}
					}
				}
				else {
					if (g_div > 2.41421356237) {
						dir[x + y] = 0;
					}
					else {
						if (g_div > 0.414213562373) {
							dir[x + y] = 3;
						}
						else {
							dir[x + y] = 2;
						}
					}
				}
			}
		}
	}
}
static void canny_non_max_suppression(sod_img * img, int *g, int *dir)
{

	int w, h, x, y, max_x, max_y;
	w = img->w;
	h = img->h;
	max_x = w;
	max_y = w * h;
	for (y = 0; y < max_y; y += w) {
		for (x = 0; x < max_x; x++) {
			switch (dir[x + y]) {
			case 0:
				if (g[x + y] > g[x + y - w] && g[x + y] > g[x + y + w]) {
					if (g[x + y] > 255) {
						img->data[x + y] = 255.;
					}
					else {
						img->data[x + y] = (float)g[x + y];
					}
				}
				else {
					img->data[x + y] = 0;
				}
				break;
			case 1:
				if (g[x + y] > g[x + y - w - 1] && g[x + y] > g[x + y + w + 1]) {
					if (g[x + y] > 255) {
						img->data[x + y] = 255.;
					}
					else {
						img->data[x + y] = (float)g[x + y];
					}
				}
				else {
					img->data[x + y] = 0;
				}
				break;
			case 2:
				if (g[x + y] > g[x + y - 1] && g[x + y] > g[x + y + 1]) {
					if (g[x + y] > 255) {
						img->data[x + y] = 255.;
					}
					else {
						img->data[x + y] = (float)g[x + y];
					}
				}
				else {
					img->data[x + y] = 0;
				}
				break;
			case 3:
				if (g[x + y] > g[x + y - w + 1] && g[x + y] > g[x + y + w - 1]) {
					if (g[x + y] > 255) {
						img->data[x + y] = 255.;
					}
					else {
						img->data[x + y] = (float)g[x + y];
					}
				}
				else {
					img->data[x + y] = 0;
				}
				break;
			default:
				break;
			}
		}
	}
}
static void canny_estimate_threshold(sod_img * img, int * high, int * low)
{

	int i, max, pixels, high_cutoff;
	int histogram[256];
	max = img->w * img->h;
	for (i = 0; i < 256; i++) {
		histogram[i] = 0;
	}
	for (i = 0; i < max; i++) {
		histogram[(int)img->data[i]]++;
	}
	pixels = (max - histogram[0]) * HIGH_THRESHOLD_PERCENTAGE;
	high_cutoff = 0;
	i = 255;
	while (high_cutoff < pixels) {
		high_cutoff += histogram[i];
		i--;
	}
	*high = i;
	i = 1;
	while (histogram[i] == 0) {
		i++;
	}
	*low = (*high + i) * LOW_THRESHOLD_PERCENTAGE;
}

static int canny_range(sod_img * img, int x, int y)
{
	if ((x < 0) || (x >= img->w)) {
		return(0);
	}
	if ((y < 0) || (y >= img->h)) {
		return(0);
	}
	return(1);
}

static int canny_trace(int x, int y, int low, sod_img * img_in, sod_img * img_out)
{
	int y_off, x_off;
	if (img_out->data[y * img_out->w + x] == 0)
	{
		img_out->data[y * img_out->w + x] = 1;
		for (y_off = -1; y_off <= 1; y_off++)
		{
			for (x_off = -1; x_off <= 1; x_off++)
			{
				if (!(y == 0 && x_off == 0) && canny_range(img_in, x + x_off, y + y_off) && (int)(img_in->data[(y + y_off) * img_out->w + x + x_off]) >= low) {
					if (canny_trace(x + x_off, y + y_off, low, img_in, img_out))
					{
						return(1);
					}
				}
			}
		}
		return(1);
	}
	return(0);
}

static void canny_hysteresis(int high, int low, sod_img * img_in, sod_img * img_out)
{
	int x, y, n, max;
	max = img_in->w * img_in->h;
	for (n = 0; n < max; n++) {
		img_out->data[n] = 0;
	}
	for (y = 0; y < img_out->h; y++) {
		for (x = 0; x < img_out->w; x++) {
			if ((int)(img_in->data[y * img_out->w + x]) >= high) {
				canny_trace(x, y, low, img_in, img_out);
			}
		}
	}
}

void sod_free_image(sod_img m)
{
	if (m.data) {
		free(m.data);
	}
}
sod_img sod_canny_edge_image(sod_img im, int reduce_noise)
{
	if (im.data && im.c == SOD_IMG_GRAYSCALE) {
		sod_img out, sobel, clean;
		int high, low, *g, *dir;
		if (reduce_noise) {
			clean = sod_gaussian_noise_reduce(im);
			if (!clean.data)return sod_make_empty_image(0, 0, 0);
		}
		else {
			clean = im;
		}
		sobel = sod_make_image(im.w, im.h, 1);
		out = sod_make_image(im.w, im.h, 1);
		g = (int*)malloc(im.w * im.h * sizeof(int));
		dir = (int*)malloc(im.w * im.h * sizeof(int));
		if (g && dir && sobel.data && out.data) {
			canny_calc_gradient_sobel(&clean, g, dir);
			canny_non_max_suppression(&sobel, g, dir);
			canny_estimate_threshold(&sobel, &high, &low);
			canny_hysteresis(high, low, &sobel, &out);
		}
		if (g)free(g);
		if (dir)free(dir);
		if (reduce_noise)sod_free_image(clean);
		sod_free_image(sobel);
		return out;
	}
	/* Make a grayscale version of your image using sod_grayscale_image() or sod_img_load_grayscale() first */
	return sod_make_empty_image(0, 0, 0);
}

sod_img sod_dilate_image(sod_img im, int times)
{
	sod_img out;
	if (im.c != SOD_IMG_GRAYSCALE) {
		/* Only grayscale or binary images */
		return sod_make_empty_image(0, 0, 0);
	}
	out = sod_make_image(im.w, im.h, im.c);
	if (out.data && im.data) {
		int x, y, w, h;
		float *srcdata = im.data;
		float *dstdata = out.data;
		float *tmp = (float*)malloc(im.w*im.h * sizeof(float));
		w = im.w;
		h = im.h;
		if (tmp) {
			while (times-- > 0) {
				for (y = 0; y < h; y++)
				{
					for (x = 0; x < w; x++)
					{
						float t;
						int x2, y2, x3, y3;

						y2 = y - 1;
						if (y2 < 0) y2 = h - 1;
						y3 = y + 1;
						if (y3 >= h) y3 = 0;

						x2 = x - 1;
						if (x2 < 0) x2 = w - 1;
						x3 = x + 1;
						if (x3 >= w) x3 = 0;


						t = srcdata[y * w + x];
						if (srcdata[y2 * w + x] > t) t = srcdata[y2 * w + x];
						if (srcdata[y3 * w + x] > t) t = srcdata[y3 * w + x];
						if (srcdata[y * w + x2] > t) t = srcdata[y * w + x2];
						if (srcdata[y * w + x3] > t) t = srcdata[y * w + x3];
						dstdata[y * w + x] = t;

					}
				}
				memcpy(tmp, dstdata, w*h * sizeof(float));
				srcdata = tmp;
			}
			free(tmp);
		}
	}
	return out;
}
#define XS (STACK[SP-3])
#define YS (STACK[SP-2])
#define ST_RETURN { SP -= 3;                \
                 switch (STACK[SP+2])    \
                 {                       \
                 case 1 : goto RETURN1;  \
                 case 2 : goto RETURN2;  \
                 case 3 : goto RETURN3;  \
                 case 4 : goto RETURN4;  \
                 default: return;        \
                 }                       \
               }
#define CALL_LabelComponent(x,y,returnLabel) { STACK[SP] = x; STACK[SP+1] = y; STACK[SP+2] = returnLabel; SP += 3; goto START; }
static void LabelComponent(uint16_t* STACK, uint16_t width, uint16_t height, float* input, sod_label_coord **output, sod_label_coord *pCord, uint16_t x, uint16_t y)
{
	STACK[0] = x;
	STACK[1] = y;
	STACK[2] = 0;  /* return - component is labeled */
	int SP = 3;
	int index;

START: /* Recursive routine starts here */

	index = XS + width * YS;
	if (input[index] == 0) ST_RETURN;   /* This pixel is not part of a component */
	if (output[index] != 0) ST_RETURN;   /* This pixel has already been labeled  */
	output[index] = pCord;

	if (pCord->xmin > XS) pCord->xmin = XS;
	if (pCord->xmax < XS) pCord->xmax = XS;
	if (pCord->ymin > YS) pCord->ymin = YS;
	if (pCord->ymax < YS) pCord->ymax = YS;

	if (XS > 0) CALL_LabelComponent(XS - 1, YS, 1);   /* left  pixel */
RETURN1:

	if (XS < width - 1) CALL_LabelComponent(XS + 1, YS, 2);   /* right pixel */
RETURN2:

	if (YS > 0) CALL_LabelComponent(XS, YS - 1, 3);   /* upper pixel */
RETURN3:

	if (YS < height - 1) CALL_LabelComponent(XS, YS + 1, 4);   /* lower pixel */
RETURN4:

	ST_RETURN;
}

static sod_label_coord * LabelImage(sod_img *pImg)
{
	sod_label_coord **apCord, *pEntry, *pList = 0;
	uint16_t width = (uint16_t)pImg->w;
	uint16_t height = (uint16_t)pImg->h;
	uint16_t* STACK;
	int labelNo = 0;
	int index = -1;
	float *input;
	uint16_t x, y;
	STACK = (uint16_t *)malloc(3 * sizeof(uint16_t)*(width*height + 1));
	if (STACK == 0) return 0;
	apCord = (sod_label_coord **)malloc(width * height * sizeof(sod_label_coord *));
	if (apCord == 0) {
		free(STACK);
		return 0;
	}
	memset(apCord, 0, width * height * sizeof(sod_label_coord *));
	input = pImg->data;
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			index++;
			if (input[index] == 0) continue;   /* This pixel is not part of a component */
			if (apCord[index] != 0) continue;   /* This pixel has already been labeled  */
												/* New component found */
			pEntry = (sod_label_coord *)malloc(sizeof(sod_label_coord));
			if (pEntry == 0) {
				goto finish;
			}
			labelNo++;
			pEntry->xmax = pEntry->ymax = -100;
			pEntry->xmin = pEntry->ymin = 2147483647;
			pEntry->pNext = pList;
			pList = pEntry;
			LabelComponent(STACK, width, height, input, apCord, pEntry, x, y);
		}
	}
finish:
	free(STACK);
	free(apCord);
	return pList;
}

typedef struct SySet SySet;
struct SySet
{
	void *pBase;               /* Base pointer */
	size_t nUsed;              /* Total number of used slots  */
	size_t nSize;              /* Total number of available slots */
	size_t eSize;              /* Size of a single slot */
	void *pUserData;           /* User private data associated with this container */
};
#define SySetBasePtr(S)           ((S)->pBase)
#define SySetBasePtrJump(S, OFFT)  (&((char *)(S)->pBase)[OFFT*(S)->eSize])
#define SySetUsed(S)              ((S)->nUsed)
#define SySetSize(S)              ((S)->nSize)
#define SySetElemSize(S)          ((S)->eSize)
#define SySetSetUserData(S, DATA)  ((S)->pUserData = DATA)
#define SySetGetUserData(S)       ((S)->pUserData)
static int SySetInit(SySet *pSet, size_t ElemSize)
{
	pSet->nSize = 0;
	pSet->nUsed = 0;
	pSet->eSize = ElemSize;
	pSet->pBase = 0;
	pSet->pUserData = 0;
	return 0;
}
static int SySetPut(SySet *pSet, const void *pItem)
{
	unsigned char *zbase;
	if (pSet->nUsed >= pSet->nSize) {
		void *pNew;
		if (pSet->nSize < 1) {
			pSet->nSize = 8;
		}
		pNew = realloc(pSet->pBase, pSet->eSize * pSet->nSize * 2);
		if (pNew == 0) {
			return SOD_OUTOFMEM;
		}
		pSet->pBase = pNew;
		pSet->nSize <<= 1;
	}
	if (pItem) {
		zbase = (unsigned char *)pSet->pBase;
		memcpy((void *)&zbase[pSet->nUsed * pSet->eSize], pItem, pSet->eSize);
		pSet->nUsed++;
	}
	return SOD_OK;
}
static void SySetRelease(SySet *pSet)
{
	if (pSet->pBase) {
		free(pSet->pBase);
	}
	pSet->pBase = 0;
	pSet->nUsed = 0;
}
int sod_image_find_blobs(sod_img im, sod_box ** paBox, int * pnBox, int(*xFilter)(int width, int height))
{
	if (im.c != SOD_IMG_GRAYSCALE || im.data == 0) {
		/* Must be a binary image */
		if (pnBox) {
			*pnBox = 0;
		}
		return SOD_UNSUPPORTED;
	}
	if (pnBox) {
		sod_label_coord *pList, *pNext;
		sod_box sRect;
		SySet aBox;
		/* Label that image */
		pList = LabelImage(&im);
		SySetInit(&aBox, sizeof(sod_box));
		while (pList) {
			pNext = pList->pNext;
			if (pList->xmax >= 0) {
				int allow = 1;
				int h = pList->ymax - pList->ymin;
				int w = pList->xmax - pList->xmin;
				if (xFilter) {
					allow = xFilter(w, h);
				}
				if (allow) {
					sRect.pUserData = 0;
					sRect.score = 0;
					sRect.zName = "blob";
					sRect.x = pList->xmin;
					sRect.y = pList->ymin;
					sRect.w = w;
					sRect.h = h;
					/* Save the box */
					SySetPut(&aBox, (const void *)&sRect);
				}
			}
			free(pList);
			pList = pNext;
		}
		*pnBox = (int)SySetUsed(&aBox);
		if (paBox) {
			*paBox = (sod_box *)SySetBasePtr(&aBox);
		}
		else {
			SySetRelease(&aBox);
		}
	}
	return SOD_OK;
}
void sod_image_draw_box_grayscale(sod_img im, int x1, int y1, int x2, int y2, float g)
{
	if (im.data) {
		int i;
		if (x1 < 0) x1 = 0;
		if (x1 >= im.w) x1 = im.w - 1;
		if (x2 < 0) x2 = 0;
		if (x2 >= im.w) x2 = im.w - 1;

		if (y1 < 0) y1 = 0;
		if (y1 >= im.h) y1 = im.h - 1;
		if (y2 < 0) y2 = 0;
		if (y2 >= im.h) y2 = im.h - 1;

		for (i = x1; i <= x2; ++i) {
			im.data[i + y1 * im.w] = g;
			im.data[i + y2 * im.w] = g;
		}
		for (i = y1; i <= y2; ++i) {
			im.data[x1 + i * im.w] = g;
			im.data[x2 + i * im.w] = g;
		}
	}
}
void sod_image_draw_box(sod_img im, int x1, int y1, int x2, int y2, float r, float g, float b)
{
	r = r / 255.;
	g = g / 255.;
	b = b / 255.;
	if (im.c == 1) {
		/* Draw on grayscale image */
		sod_image_draw_box_grayscale(im, x1, y1, x2, y2, b * 0.114 + g * 0.587 + r * 0.299);
		return;
	}
	if (im.data) {
		int i;
		if (x1 < 0) x1 = 0;
		if (x1 >= im.w) x1 = im.w - 1;
		if (x2 < 0) x2 = 0;
		if (x2 >= im.w) x2 = im.w - 1;

		if (y1 < 0) y1 = 0;
		if (y1 >= im.h) y1 = im.h - 1;
		if (y2 < 0) y2 = 0;
		if (y2 >= im.h) y2 = im.h - 1;

		for (i = x1; i <= x2; ++i) {
			im.data[i + y1 * im.w + 0 * im.w*im.h] = r;
			im.data[i + y2 * im.w + 0 * im.w*im.h] = r;

			im.data[i + y1 * im.w + 1 * im.w*im.h] = g;
			im.data[i + y2 * im.w + 1 * im.w*im.h] = g;

			im.data[i + y1 * im.w + 2 * im.w*im.h] = b;
			im.data[i + y2 * im.w + 2 * im.w*im.h] = b;
		}
		for (i = y1; i <= y2; ++i) {
			im.data[x1 + i * im.w + 0 * im.w*im.h] = r;
			im.data[x2 + i * im.w + 0 * im.w*im.h] = r;

			im.data[x1 + i * im.w + 1 * im.w*im.h] = g;
			im.data[x2 + i * im.w + 1 * im.w*im.h] = g;

			im.data[x1 + i * im.w + 2 * im.w*im.h] = b;
			im.data[x2 + i * im.w + 2 * im.w*im.h] = b;
		}
	}
}
void sod_image_draw_bbox_width(sod_img im, sod_box bbox, int width, float r, float g, float b)
{
	int i;
	for (i = 0; i < width; i++) {
		sod_image_draw_box(im, bbox.x + i, bbox.y + i, (bbox.x + bbox.w) - i, (bbox.y + bbox.h) - i, r, g, b);
	}
}

void sod_image_blob_boxes_release(sod_box * pBox)
{
	free(pBox);
}

unsigned char * sod_image_to_blob(sod_img im)
{
	unsigned char *data = 0;
	int i, k;
	if (im.data) {
		data = (unsigned char*)calloc(im.w*im.h*im.c, sizeof(unsigned char));
		if (data) {
			for (k = 0; k < im.c; ++k) {
				for (i = 0; i < im.w*im.h; ++i) {
					data[i*im.c + k] = (unsigned char)(255 * im.data[i + k * im.w*im.h]);
				}
			}
		}
	}
	return data;
}

void sod_image_free_blob(unsigned char *zBlob)
{
	free(zBlob);
}

/*sod_img_save_as_png(sod_img input, const char * zpath)
{
unsigned char *zpng = sod_image_to_blob(input);
int rc;
if (zpng == 0) {
return sod_outofmem;
}
rc = stbi_write_png(zpath, input.w, input.h, input.c, (const void *)zpng, input.w * input.c);
sod_image_free_blob(zpng);
return rc ? sod_ok : sod_ioerr;
}*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
* Frontal License Plate detection without deep-learning. Only image processing code.
*/
static int filter_cb(int width, int height)
{
	/* A filter callback invoked by the blob routine each time
	* a potential blob region is identified.
	* We use the `width` and `height` parameters supplied
	* to discard regions of non interest (i.e. too big or too small).
	*/
	if ((width > 300 || height > 100) || width < 45 || height < 55) {
		/* Ignore small or big boxes (You should take in consideration
		* U.S plate size here and adjust accordingly).
		*/
		return 0; /* Discarded region */
	}
	return 1; /* Accepted region */
}

#define WIDTH  640
#define HEIGHT 570
int main(int argc, char *argv[])
{
	
	FILE *fp;
	float *data;
	float buff;
	data = (float*)malloc(WIDTH * HEIGHT * sizeof(float));

	fp = fopen("00191_inp.txt", "r");
	for (int i = 0; i < HEIGHT; i++) {
		for (int j = 0; j < WIDTH; j++)
		{
			fscanf(fp, "%f", &buff);
			data[i * WIDTH + j] = buff;
		}
	}
	fclose(fp);

	sod_img zInput;
	zInput.h = HEIGHT;
	zInput.w = WIDTH;
	zInput.c = 1;
	zInput.data = data;

	sod_img imgIn = sod_grayscale_image(zInput);
	if (imgIn.data == 0) {
		/* Invalid path, unsupported format, memory failure, etc. */
		puts("Cannot load input image..exiting");
		return 0;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////xong
	/* A full color copy of the input image so we can draw rose boxes
	* marking the plate in question if any.
	*/
	//sod_img imgCopy = sod_img_load_color(zInput);
	/* Obtain a binary image first */
	sod_img binImg = sod_threshold_image(imgIn, 128);
	
	/*
	* Perform Canny edge detection next which is a mandatory step
	*/
	//sod_img cannyImg = sod_canny_edge_image(binImg, 1/* Reduce noise */);
	sod_img cannyImg = sod_canny_edge_image(binImg, 1/* Reduce noise */);

	/*
	* Dilate the image say 12 times but you should experiment
	* with different values for best results which depend
	* on the quality of the input image/frame. */
	sod_img dilImg = sod_dilate_image(cannyImg, 12);
	/* Perform connected component labeling or blob detection
	* now on the binary, canny edged, Gaussian noise reduced and
	* finally dilated image using our filter callback that should
	* discard small or large rectangle areas.
	*/
	sod_box *box = 0;
	int i, nbox;
	sod_image_find_blobs(dilImg, &box, &nbox, filter_cb);
	/* Draw a box on each potential plate coordinates */
	for (i = 0; i < nbox; i++) {
		//sod_image_draw_bbox_width(imgCopy, box[i], 5, 255., 0, 225.); // rose box
		sod_image_draw_bbox_width(imgIn, box[i], 5, 255., 0, 225.); // rose box

	}
	sod_image_blob_boxes_release(box);

	///////
	FILE *fp_out;
	fp_out = fopen("00191_out.txt", "w+");
	for (int i = 0; i < HEIGHT; i++) {
		for (int j = 0; j < WIDTH; j++)
		{
			fprintf(fp_out, "%d ", (int)imgIn.data[i * WIDTH + j]);
		}
		fprintf(fp_out, " \n");
	}
	fclose(fp_out);
	////////////////////////////
	/* Finally save the output image to the specified path */
	//sod_img_save_as_png(imgCopy, zOut);
	/* Cleanup */
	sod_free_image(imgIn);
	sod_free_image(cannyImg);
	sod_free_image(binImg);
	sod_free_image(dilImg);

	free(data);
	return 0;
}