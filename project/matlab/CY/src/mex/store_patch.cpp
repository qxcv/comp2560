#include "mex.h"
#include <string>
#include <lmdb.h>
#include <sys/stat.h>
#include <sstream>
#include "caffe/proto/caffe.pb.h"

using std::string;
using std::ostringstream;
using caffe::Datum;

// print and throw a Mex error
inline void mex_error(const std::string &msg) {
  mexErrMsgTxt(msg.c_str());
}

// print Warning message
inline void mex_warn(const std::string &msg) {
  mexWarnMsgTxt(msg.c_str());
}

// print Info message
inline void mex_info(const std::string &msg) {
  mexPrintf(msg.c_str());
}

const int BATCH_N = 1000;
bool MatToDatum(const uint8_t* im, const int label,
        const int height, const int width, Datum* datum, const bool is_color = true);

void crop_patch(const mxArray *mx_iminfo, const mxArray *mx_label,
        const mxArray *mx_psize, mxArray** mx_cpatch);

// matlab entry point
// Cells of property = store_patch(imdata, labels, psize, database name);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs != 4)
    mex_error("Wrong number of inputs");
  if (nlhs != 0)
    mex_error("Wrong number of outputs");
  
  // get imdata
  const mxArray *mx_imdata = prhs[0];
  if (!mxIsCell(mx_imdata))
    mex_error("Invalid input: imdata");
  
  // get labels
  const mxArray *mx_labdata = prhs[1];
  if (!mxIsCell(mx_labdata))
    mex_error("Invalid input: labels");
  
  size_t N = mxGetNumberOfElements(mx_imdata);
  if (mxGetNumberOfElements(mx_labdata) != N)
    mex_error("imdata and labels should have the same number of elements");
  
  // get psize
  const mxArray *mx_psize = prhs[2];
  if (!mxIsDouble(mx_psize))
    mex_error("Invalid input: psize");
  
  // get filename
  char *c_dbname = mxArrayToString(prhs[3]);
  if (!c_dbname)
    mex_error("Invalid input: dbname");
  const string dbname(c_dbname);
  mxFree(static_cast<void*>(c_dbname));
  // -------------- open database -----------
  // Open new db
  // lmdb
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;
  
  mex_info("Opening lmdb " + dbname + "\n");
  int status = mkdir(dbname.c_str(), 0744);
  if (status != 0) {
    mex_error("mkdir " + dbname + " failed");
  }
  if (mdb_env_create(&mdb_env) != MDB_SUCCESS)
    mex_error("mdb_env_create failed");
  if (mdb_env_set_mapsize(mdb_env, 1099511627776) != MDB_SUCCESS)  // 1TB
    mex_error("mdb_env_set_mapsize failed");
  if (mdb_env_open(mdb_env, dbname.c_str(), 0, 0664) != MDB_SUCCESS)
    mex_error("mdb_env_open failed");
  if (mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn) != MDB_SUCCESS)
    mex_error("mdb_txn_begin failed");
  if (mdb_open(mdb_txn, NULL, 0, &mdb_dbi) != MDB_SUCCESS)
    mex_error("mdb_open failed");
  Datum datum;
  int count = 0;
  bool data_size_initialized = false;
  int data_size;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  mxArray *mx_cpatch = NULL;
  
  for (size_t i = 0; i < N; i++) {
    const mxArray *mx_iminfo = mxGetCell(mx_imdata, i);
    const mxArray *mx_labinfo = mxGetCell(mx_labdata, i);
    char *c_imname = mxArrayToString(mxGetField(mx_iminfo, 0, "im"));	// mx_iminfo is 1x1 structure
    const string imname(c_imname);
    mxFree(static_cast<void*>(c_imname));
    const double r_degree = mxGetScalar(mxGetField(mx_iminfo, 0, "r_degree"));
    const bool isflip = mxGetScalar(mxGetField(mx_iminfo, 0, "isflip")) != 0;
    // crop_patch is implemented in Matlab
    crop_patch(mx_iminfo, mx_labinfo, mx_psize, &mx_cpatch);
    // store
    size_t P_N = mxGetNumberOfElements(mx_cpatch);
    for (size_t j = 0; j < P_N; j++) {
      const mxArray* mx_patch = mxGetField(mx_cpatch, j, "patch");
      const mxArray* mx_label = mxGetField(mx_cpatch, j, "label");
      const mxArray* mx_rngkey = mxGetField(mx_cpatch, j, "rngkey");
      if (mxGetClassID(mx_patch) != mxUINT8_CLASS)
        mex_error("patch should be uint8");
      if (mxGetClassID(mx_label) != mxINT32_CLASS)
        mex_error("label should be int32");
      if (mxGetClassID(mx_rngkey) != mxINT32_CLASS)
        mex_error("rngkey should be int32");
      
      const int rngkey = static_cast<int>(mxGetScalar(mx_rngkey));
      const int label = static_cast<int>(mxGetScalar(mx_label));
      if (rngkey >= 1e9 || rngkey <= 0) {
        mex_error("rngkey should be [1, 1e9)");
      }
      const bool is_color = mxGetNumberOfDimensions(mx_patch) == 3;
      const mwSize* dims = mxGetDimensions(mx_patch);
      const int height = dims[1];
      const int width = dims[0];  			// width is fastest in caffe
      const uint8_t* im = static_cast<uint8_t*>(mxGetData(mx_patch));
      
      snprintf(key_cstr, kMaxKeyLength, "%09i_%lu_%lu_%s_%f_%i",
              rngkey, i, j, imname.c_str(), r_degree, isflip);
      string keystr(key_cstr);
      if (!MatToDatum(im, label, height, width, &datum, is_color))
        mex_error("Read " + keystr + " failed");
      
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const string& data = datum.data();
        if (data.size() != data_size) {
          ostringstream oss;
          oss << "Incorrect data field size " << data.size();
          mex_error(oss.str());
        }
      }
      // ------ sequential --------
      string value;
      datum.SerializeToString(&value);
      // ------- store in db ---------
      mdb_data.mv_size = value.size();
      mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
      mdb_key.mv_size = keystr.size();
      mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
      if ((status = mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0)) != MDB_SUCCESS) {
        ostringstream oss;
        oss << "mdb_put failed, status = " << status;
        mex_error(oss.str());
      }
      if (++count % BATCH_N == 0) {
        if ((status = mdb_txn_commit(mdb_txn)) != MDB_SUCCESS) {
          ostringstream oss;
          oss << "mdb_txn_commit failed, status = " << status;
          mex_error(oss.str());
        }
        if ((status = mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn)) != MDB_SUCCESS) {
          ostringstream oss;
          oss << "mdb_txn_begin failed, status = " << status;
          mex_error(oss.str());
        }
        ostringstream oss;
        oss << "Processed " << count << " patches." << std::endl;
        mex_info(oss.str());
      }
    }	// for patches
  }	// for images
  // write the last batch
  if (count % BATCH_N != 0) {
    if (mdb_txn_commit(mdb_txn) != MDB_SUCCESS)
      mex_error("mdb_txn_commit failed");
    mdb_close(mdb_env, mdb_dbi);
    mdb_env_close(mdb_env);
    ostringstream oss;
    oss << "Processed " << count << " patches." << std::endl;
    mex_info(oss.str());
  }
}

bool MatToDatum(const uint8_t* im, const int label,
        const int height, const int width, caffe::Datum* datum, const bool is_color) {
  int num_channels = (is_color ? 3 : 1);
  datum->set_channels(num_channels);
  datum->set_height(height);
  datum->set_width(width);
  
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  datum_string->assign(reinterpret_cast<const char *>(im), height*width*num_channels);
  return true;
}

void crop_patch(const mxArray *mx_iminfo, const mxArray *mx_label,
        const mxArray *mx_psize, mxArray** mx_cpatch) {
  
  mxArray *prhs[3] = { const_cast<mxArray*>(mx_iminfo),
  const_cast<mxArray*>(mx_label),
  const_cast<mxArray*>(mx_psize) };
  // crop_patch will not modify data
  if (*mx_cpatch) {
    // clear
    mxDestroyArray(*mx_cpatch);
  }
  int status = mexCallMATLAB(1, mx_cpatch, 3, prhs, "crop_patch");
  if (status) {// failed
    mex_error("Call crop_patch failed");
  }
}


