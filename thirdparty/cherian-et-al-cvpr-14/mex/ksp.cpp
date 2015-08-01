// Dijkstra's shortest path finding algorithm in mex. 
// implemented by Anoop Cherian, email:anoop.cherian@inria.fr.
// coded on 11th March 2013.

#include <math.h>
#include "matrix.h"
#include "mex.h"

void mexPrintMatrix(mxArray *matptr);
int populate_affinity_matrices(const mxArray*, double**, int *);	
int	find_shortest_path(const mxArray*, mxArray**, mxArray**);
int get_max_states(const mxArray *cellarrayptr, int numcells);
mxArray* augment_cell_array(const mxArray* cellarrayptr);

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


// get the maximum of the dimensions in the input cell array.
int get_max_states(const mxArray *cellarrayptr, int numcells)
{
	unsigned int d=0, m, n;
	mxArray* cellptr;
	for (int i=0; i<numcells; i++)
	{
		cellptr = mxGetCell(cellarrayptr, i);
		if (NULL != cellptr)
		{
			m = mxGetM(cellptr);
			n = mxGetN(cellptr);
			m = (m>=n)?m:n;
			if (m > d)		
			{
				d = m;		
			}
		}
	}
	return d;
}


// you need to free the mxArray* returned.
mxArray* augment_cell_array(const mxArray* cellarrayptr, int *max_dims)
{
	mxArray *newC, *cellptr, *zerosD1, *zerosD2, *newcellptr;
	size_t numcells;	
	int d;
	
	numcells = mxGetNumberOfElements(cellarrayptr);
	newC= mxCreateCellMatrix((mwSize)numcells+2, 1); // new cell array. 
	d = get_max_states(cellarrayptr, numcells);
	if (d<=0)
	{
		return NULL;
	}

	zerosD1 = mxCreateDoubleMatrix(d,d, mxREAL); // initialize the zeros matrix.
	zerosD2 = mxCreateDoubleMatrix(d,d, mxREAL); // final zerosD.

	mxSetCell(newC, 0, zerosD1); // first item	
	for (unsigned int i=0; i<numcells; i++) // copy the input cells.
	{
		cellptr = mxGetCell(cellarrayptr, i);
		newcellptr = mxDuplicateArray(cellptr); // duplicate the matrix for the new cell array.
		mxSetCell(newC, i+1, newcellptr);
	}
	mxSetCell(newC, numcells+1, zerosD2); // last item.
	*max_dims = d;
	return newC;
}

// intialize dist.
template <class T>
void initialize_matrix(T* dist_ptr, int d, int n, T val)
{
	for (int i=0; i<d; i++)
			for (int j=0; j<n; j++)			
				*(dist_ptr + i*n+j) = val; //initial distance is inf.
	return;
}

// get min value of nth column
double get_min_value(double *dist_ptr, int d, int n, double *cost, int* p)
{
	double tmp_val;
	*cost = mxGetInf();
	for (int i=0; i<d; i++)
	{
		tmp_val = *(dist_ptr + (n-1)*d + i); // last column of dist_ptr ?
		if (tmp_val < *cost) //last cost.
		{
			*cost = tmp_val;
			*p = i;
		}
	}
	return *cost;
}

// shortest path finding algorithm
int find_shortest_path(const mxArray* cellarrayptr, mxArray** dist_out, mxArray** path_out)
{	
	int d, M, N;  // max dims.
	double inf, s; //infinity
	mxArray *C, *prev, *dist, *path, *cell;
	double *dist_ptr, *cellptr;
	int *prev_ptr;	
	
	C = augment_cell_array(cellarrayptr, &d);		
	size_t n = mxGetNumberOfElements(C);
	//mexPrintf("cell array size = %d\n", n);

	// what is infinity?
	inf = mxGetInf();

	// distance matrix.
	dist_ptr = (double*)mxCalloc(n*d, sizeof(double));
	initialize_matrix(dist_ptr, d, n, inf); // intialize dist to inf.

	// 'previous' variable.
	prev_ptr = (int*)mxCalloc(n*d, sizeof(int)); // 		
	initialize_matrix(prev_ptr, d, n, -1);

	// now lets start the shortest path algo
	for (int i=0; i<d; i++)
		*(dist_ptr + i) = 0; // set dist(:,1)=0;

	// test
	//mexPrintMatrix(dist_ptr, n, d, " %f ");
	//mexPrintMatrix(prev_ptr, n, d, " %d ");
	
	for (int k=1; k<n; k++)
	{
		cell = mxGetCell(C, k);
		cellptr = mxGetPr(cell);
		M = mxGetM(cell);
		N = mxGetN(cell);
		for (int i=0; i<M; i++) // ydim
		{			
			for (int j=0; j<N; j++) //xdim
			{
	//			mexPrintf("dist_ptr + i*d + k-1 = %f cellptr + i*N + j=%f\n", *(dist_ptr + i + (k-1)*n), *(cellptr + i + j*M));
				s = *(dist_ptr + (k-1)*d + i) + *(cellptr + j*M+i);
				if (s < *(dist_ptr + k*d + j))
				{
					*(dist_ptr + k*d + j) = s;
					*(prev_ptr + k*d + j) = i;		
				}
			}
		}
		//mexPrintMatrix(dist_ptr, n, d, " %f ");
		//mexPrintMatrix(prev_ptr, n, d, " %d ");
	}

	// how to access the matrix.
	/*for (int j=0; j<ydim; j++)
	{
		for (int k=0; k<xdim; k++)
		{
			mexPrintf(" %lf ", *(matcontentptr + k*ydim + j));
		}
		mexPrintf("\n");
	}*/

	// now lets get the shortest path.
	//mxArray* s_matrix = mxCreateNumericMatrix(n,1, mxINT32_CLASS, mxREAL);
	int *s_ptr = (int*)mxCalloc(n, sizeof(int)); // will hold the path
	double *cost = (double*)mxCalloc(n, sizeof(double));
	int p = -1; //prev ptr	

	// find the min values of cost and p
	get_min_value(dist_ptr, d, n, &cost[n-1], &p); // find the min value of dist(:,n).

	bool failed = false;
	if (p == -1) // no shortest paths
	{		
		failed = true;		
	}
	else
	{
		// find the path
		for (int k=n-1; k>=0; k--)
		{
			s_ptr[k] = p;
			if (p==-1) //at any time in the loop.
			{
				mexPrintf("failed!\n");
				failed = true;				
				break;
			}

			cost[k] = *(dist_ptr + k*d + p);
			p = *(prev_ptr + k*d + p);
		}		
	}

	// now lets make the out arrays
	mxArray *cost_out = mxCreateDoubleMatrix(1, n-1, mxREAL);
	double *cost_out_ptr  = mxGetPr(cost_out);
	mxArray *p_out = mxCreateNumericMatrix(1, n-1, mxINT32_CLASS, mxREAL);
	int *p_out_ptr = (int*)mxGetData(p_out);
	if (!failed)
	{
		for (int i=0; i<n-1; i++)
		{
			cost_out_ptr[i] = cost[i];
			p_out_ptr[i] = s_ptr[i] + 1; // 1-indexed.
		}
	}
	else
	{
		initialize_matrix(cost_out_ptr, 1, n-1, inf);		
	}
	*dist_out = cost_out;
	*path_out = p_out;		

	// now free all the temp matrices
	mxFree(dist_ptr);
	mxFree(prev_ptr);
	mxFree(s_ptr);
	mxFree(cost);

	mxDestroyArray(C);
	return 0;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mxArray *dist_out, *path_out;
	const mxArray *cellarrayptr;	
	int numouts;	
		
	numouts = nlhs;	
	if ((nlhs != 2) || (nrhs != 1))
	{
		mexPrintf("Error: Number of outputs must be 2 and Number of inputs must be 1\n");
		return;
	}

	if (!mxIsCell(prhs[0]))
	{
		mexPrintf("Error: input must be of celltype\n");
		return;
	}
	
	// now lets associate the input and outputs.	
	cellarrayptr = prhs[0];	
	
	find_shortest_path(cellarrayptr, &dist_out, &path_out);
	
	plhs[0] = dist_out;
	plhs[1] = path_out;
	
	return;			
}

