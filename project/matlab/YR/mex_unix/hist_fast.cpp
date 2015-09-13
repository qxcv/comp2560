// Anoop's implementation of the fast histogram.
// implemented on 17th March 2013.
// anoop.cherian@inria.fr
#include <math.h>
#include <string.h>
#include <mex.h>
#include <matrix.h>
//overloaded function
template <class T>
void  mexPrintMatrix(T *matptr, int xdim, int ydim, const char *str)
{			
	for (int j=0; j<ydim; j++)
	{
		for (int k=0; k<xdim; k++)
		{			
			mexPrintf( str, *(matptr + k*ydim + j));
		}
		mexPrintf("\n");
	}
	mexPrintf("\n\n");
	return;
}

// compute the min val of an array or a matrix.
double find_min(double *dat, const mwSize *sz, double sc)
{
	double min_val = mxGetInf();
	double val;	
	for (int x=0; x<sz[1]; x++)
	{
		for (int y=0; y<sz[0]; y++)
		{
			val = sc*(*(dat + x*sz[0] + y)); // we use the same code for max and min. sc={-1,+1}			
			if (val < min_val)			
			{
				min_val = val;				
			}
		}
	}
	return min_val;
}

// compute the diff function for a vector x.
double *get_diff(double *x, int x_len)
{
	double *xdiff = (double*)mxCalloc(x_len-1, sizeof(double));
	for (int i=1; i<x_len; i++)
		xdiff[i-1] = *(x+i)-*(x+i-1);

	return xdiff;
}

// debugging function
void disp_vector(double *x, int x_len)
{
	for (int i=0; i<x_len; i++)
		mexPrintf(" %f ", x[i]);
	mexPrintf("\n");
	return;
}

// this routine will call the mex eps function
double* get_epsxx(double*xx, int x_len)
{
	int nlhs=1, nrhs=1;
	mxArray *prhs[1], *plhs[1];
	double *epsxx; 

	prhs[0] = mxCreateDoubleMatrix(1, x_len, mxREAL);
	memcpy(mxGetPr(prhs[0]), xx, x_len*sizeof(double));
	mexCallMATLAB(nlhs, plhs, nrhs, prhs, "eps");

	epsxx = (double*)mxCalloc(x_len, sizeof(double));
	memcpy(epsxx, mxGetPr(plhs[0]), x_len*sizeof(double));	
	mxDestroyArray(plhs[0]);
	
	return epsxx;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	const mxArray* x_in, *y_in;
	double *x, *y, miny, maxy, *xx;
	const mwSize *sz, *dims;
	mxArray* mex_edges,	*mex_dims, *nn;
	double *edges;		
	int x_len, m, n, numdims;

	y_in = prhs[0]; 
	x_in = prhs[1];

	// get the data ptrs
	y = mxGetPr(y_in);	
	x = mxGetPr(x_in);
	sz = mxGetDimensions(y_in);		

	dims = mxGetDimensions(x_in);
	numdims = mxGetNumberOfDimensions(x_in);		
/*	if (numdims !=2)
	{
		mexPrintf("dims of x_in should be 2\n");
		return;
	}
*/
	if (dims[0]==1 && dims[1]==1)
	{
		/*if (!mxIsNumeric(x_in))
		{
			mexPrintf("Error:x_in if a scalar, has to be an integer\n");
			return;
		}*/
		x_len = int(*x);
	}
	else
	{
		x_len = (dims[0]>dims[1])? dims[0]:dims[1];	
	}
	
	/*
	if (x_len == 0)
	{
		mexPrintf("Error: length of x_in should not be zero\n");
		return;
	}
	*/

	// compute the range of y.
	miny = find_min(y, sz, 1);
	maxy = -find_min(y,sz, -1); //find_max(y, sz);	

	if ((dims[0]==1) && (dims[1]==1))
	{					
		double binwidth;
		if (miny==maxy)
		{
			miny = miny - floor(double(x_len)/2) - 0.5;
			maxy = maxy + ceil(double(x_len)/2) - 0.5;
		}		
		binwidth = (maxy-miny)/double(x_len);		
		xx = (double*)mxCalloc(x_len+1, sizeof(double)); //x_len=+1 bcos 0:x			

		for (int i=0; i<x_len+1; i++)	
		{
			*(xx+i) = miny + binwidth*double(i);		
		}
		*(xx+x_len) = maxy;
		x_len = x_len + 1;		 
	}
	else
	{
		double *binwidth, *xxextra, *xxdiff;
		xx = x;				
		binwidth = (double*)mxCalloc(x_len, sizeof(double));
		xxextra = (double*)mxCalloc(x_len+1, sizeof(double));
		xxdiff = get_diff(xx, x_len);		

		for (int i=0; i<x_len-1; i++)		
		{
			binwidth[i] = *(xxdiff + i);		
		}
		binwidth[x_len-1] = 0;		

		*xxextra = *xx - binwidth[0]/2;
		for (int j=0; j<x_len; j++)		
		{
			*(xxextra+j+1) = *(xx+j) + *(binwidth+j)/2;		
		}

		xx = xxextra; // xx is just x which is prhs[1]. so no mem leak.
		x_len=x_len+1;  // xx is now xx+1 as xxextra is xx+1

		//*xx = fmin(*xx, miny);
		*xx = (*xx>miny) ? miny : *xx;		

		//*(xx+x_len-1) = fmax(*(xx+x_len-1), maxy);
		*(xx+x_len-1) = (*(xx+x_len-1)>maxy) ? *(xx+x_len-1) : maxy;

		mxFree(binwidth);
		mxFree(xxdiff);
	}

	// get eps_xx. this is important!
	double *eps_xx = get_epsxx(xx, x_len);

	edges = (double*)mxCalloc(x_len+1, sizeof(double));
	*edges = -mxGetInf();	
	for (int j=1; j<x_len+1; j++)
	{
		edges[j] = xx[j-1] + eps_xx[j-1];
	}	
		
	double *mex_edges_ptr;
	mex_edges = mxCreateDoubleMatrix(1, x_len+1, mxREAL);
	mex_edges_ptr = mxGetPr(mex_edges);
	memcpy(mex_edges_ptr, edges, (x_len+1)*sizeof(double));
		
	mex_dims = mxCreateNumericMatrix(1, 1, mxUINT32_CLASS, mxREAL);
	int *mex_dims_ptr = (int*)mxGetData(mex_dims);
	*mex_dims_ptr = 1; // dims over which the histc is taken
	
	// now lets call the histc matlab fn
	//int mexCallMATLAB(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[], const char *functionName);
	mxArray* plhs_histc[1];
	mxArray* prhs_histc[3]={(mxArray*)y_in, mex_edges, mex_dims};		
	mexCallMATLAB(1, plhs_histc, 3, prhs_histc, "histc");	

	// now lets analyze the output of histc.
	nn = plhs_histc[0];
	m  = mxGetM(nn);
	n  = mxGetN(nn);			

	// combine first bin with 2nd bin and last bin
	double* nn_ptr = mxGetPr(nn);
	*(nn_ptr+1) = *(nn_ptr+1) + *(nn_ptr);
	*(nn_ptr+m-2) = *(nn_ptr+m-2) + *(nn_ptr+m-1);			

	mxArray *nn_out=mxCreateDoubleMatrix(1,m-2, mxREAL);
	double* nn_out_ptr = mxGetPr(nn_out);
	for (int i=1; i<m-1; i++)	
		*(nn_out_ptr + i-1) = *(nn_ptr+i);	
	
	// and we are done!
	plhs[0] = nn_out;

	// now free all the extra allocations
	mxDestroyArray(mex_edges);
	mxDestroyArray(mex_dims);
	mxDestroyArray(nn);
	mxFree(edges);		
	mxFree(xx);
	mxFree(eps_xx);
	return;
}
