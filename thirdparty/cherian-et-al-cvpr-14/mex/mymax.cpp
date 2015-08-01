// Dijkstra's shortest path finding algorithm in mex. 
// implemented by Anoop Cherian, email:anoop.cherian@inria.fr.
// coded on 11th March 2013.

#include <math.h>
#include "matrix.h"
#include "mex.h"


void  mexPrintMatrix(mxArray *matptr)
{
	int xdim, ydim; 
	double *matcontentptr; 
	const mwSize *dims;

	matcontentptr = mxGetPr(matptr);
	dims = mxGetDimensions(matptr);
	mexPrintf("matrix : numrows = %d numcols=%d\n", (int)dims[0], (int)dims[1]);
	xdim = (int)dims[1]; ydim=(int)dims[0];
	for (int j=0; j<ydim; j++)
	{
		for (int k=0; k<xdim; k++)
		{
			mexPrintf(" %lf ", *(matcontentptr + k*ydim + j));
		}
		mexPrintf("\n");
	}
	mexPrintf("\n\n");
	return;
}

/* get min value of nth column*/
double compute_max(double *M, int d, int n)
{
	double maxval = 0;//-mxGetInf();
	double val;
	for (int i=0; i<d; i++) {
	    for (int j=0; j<n; j++) 
	    {
		val = *(M+n*i+j);
		if (fabs(val) > fabs(maxval))		
		{
		    maxval = val;		
		}
	    }
	}
	return maxval;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{		
	int numouts = nlhs;	
	int d,n; /* holds the matrix dimensions*/
	double *M; /* holds the matrix.*/
	const mxArray *Mptr;
	double maxval;
	mxArray* outval;
/*
	if ((nlhs != 1) || (nrhs != 1))
	{
		mexPrintf("Usage: val = mymax(M);\n");
		return;
	}
*/
	/* now lets associate the input and outputs.	*/
	Mptr = prhs[0];	
        d = mxGetM(Mptr); n = mxGetN(Mptr); M = mxGetPr(Mptr); 
	maxval = compute_max(M, d, n);
	outval = mxCreateDoubleScalar(maxval);
	plhs[0] = outval;
	return;			
}

