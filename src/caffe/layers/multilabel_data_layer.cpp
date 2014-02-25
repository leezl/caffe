// Copyright 2013 Yangqing Jia
//edits: liesl wigand
/*
This began as the datalayer, but is being altered to use multilabel data.

NOTE: The output layers should also handle multilabel data.
NOTE: Only provide multi_label data in the datum for this class. We cannot 
change single label without knowing the number of classes, which we don't. yet.
*/

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
void* MultiLabelDataLayerPrefetch(void* layer_pointer) {
  MultiLabelDataLayer<Dtype>* layer = reinterpret_cast<MultiLabelDataLayer<Dtype>*>(layer_pointer);
  Datum datum;
  // so...where is this set to the correct size...
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  const Dtype scale = layer->layer_param_.scale();
  const int batchsize = layer->layer_param_.batchsize();
  const int cropsize = layer->layer_param_.cropsize();
  const bool mirror = layer->layer_param_.mirror();

  if (mirror && cropsize == 0) {
    LOG(FATAL) << "Current implementation requires mirror and cropsize to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = layer->datum_channels_;
  const int height = layer->datum_height_;
  const int width = layer->datum_width_;
  const int size = layer->datum_size_;
  const int label_size = layer->datum_label_size_;
  const Dtype* mean = layer->data_mean_.cpu_data();

  for (int itemid = 0; itemid < batchsize; ++itemid) {
    // get a blob
    datum.ParseFromString(layer->iter_->value().ToString());
    const string& data = datum.data();

    if (cropsize) { //INVESTIGATE CROPPING
      //CHECK(data.size()) << "Image cropping only supports uint8 data";
      //how much height and width we cut off, and where
      int h_off, w_off;
      // We only do random crop when we do training, and sometimes not even then
      if ( Caffe::phase() == Caffe::TRAIN ) {
        // a random value in range 0 to (height - cropsize) -> so where to begin crop 
        //if image is 640 x 480, crop 200 -> we can get a cropped version anywhere in the image from this
        h_off = rand() % (height - cropsize);//amount of height minus crop -> space we don't use
        w_off = rand() % (width - cropsize);
      } else {
        h_off = (height - cropsize) / 2;
        w_off = (width - cropsize) / 2;
      }
      //if we have int data
      if (data.size()) {
        //only 50% of time mirror
        if (mirror && rand() % 2) {
          // Copy mirrored version
          for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < cropsize; ++h) {
              for (int w = 0; w < cropsize; ++w) {
                top_data[((itemid * channels + c) * cropsize + h) * cropsize
                         + cropsize - 1 - w] =
                    (static_cast<Dtype>(
                        (uint8_t)data[(c * height + h + h_off) * width
                                      + w + w_off])
                      - mean[(c * height + h + h_off) * width + w + w_off])
                    * scale;
              }
            }
          }
        } else {
          // Normal copy
          for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < cropsize; ++h) {
              for (int w = 0; w < cropsize; ++w) {
                top_data[((itemid * channels + c) * cropsize + h) * cropsize + w]
                    = (static_cast<Dtype>(
                        (uint8_t)data[(c * height + h + h_off) * width
                                      + w + w_off])
                       - mean[(c * height + h + h_off) * width + w + w_off])
                    * scale;
              }
            }
          }
        }
      } else {
        //float cropping...NEEDS TESTING
        if (mirror && rand() % 2) {
          // Copy mirrored version
          for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < cropsize; ++h) {
              for (int w = 0; w < cropsize; ++w) {
                top_data[((itemid * channels + c) * cropsize + h) * cropsize
                         + cropsize - 1 - w] =
                    (datum.float_data((c * height + h + h_off) * width
                                      + w + w_off)
                      - mean[(c * height + h + h_off) * width + w + w_off])
                    * scale;
              }
            }
          }
        } else {
          // Normal copy
          for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < cropsize; ++h) {
              for (int w = 0; w < cropsize; ++w) {
                top_data[((itemid * channels + c) * cropsize + h) * cropsize + w]
                    = (datum.float_data((c * height + h + h_off) * width
                                      + w + w_off)
                       - mean[(c * height + h + h_off) * width + w + w_off])
                    * scale;
              }
            }
          }
        }
      } //end float cropping
    } else {
      //no cropping
        // we will prefer to use data() first, and then try float_data()
        if (data.size()) {
          //int only data
          for (int j = 0; j < size; ++j) {
            top_data[itemid * size + j] =
                (static_cast<Dtype>((uint8_t)data[j]) - mean[j]) * scale;
          }
        } else {
          //float data
          for (int j = 0; j < size; ++j) {
            top_data[itemid * size + j] =
                (datum.float_data(j) - mean[j]) * scale;
          }
        }
      }
    
    //copy label into top
    //top_label[itemid] = datum.label(); //old way
    //check that we do not have mismatched labels
      //following really should never happen
    if (label_size != datum.multi_label_size() && datum.multi_label_size() !=0) {
      LOG(FATAL) << "Our number of labels does not match up: it must be 1 in label, "
        << "Or many in multilabel, and the layers label size must match one of these. "
        <<label_size<<",  "<<datum.multi_label_size();
    }
    // So: check f single label or many
    //After the checks in setup, we know we have multi_label data
    for (int j = 0; j < label_size; ++j) {
        top_label[itemid * label_size + j] =
            (datum.multi_label(j));
    }
    /////////////////Following possible later
    /*
    //IF we knew the total number of classes, and set top_label appropriately:
    if (datum.has_label() && !(datum.multi_label_size()>0)){ //we have single label
      for (int j = 0; j < NUMCLASSES; ++j) {
        if(j==datum.label()) { //assuming label is index of value still
          top_label[itemid * label_size + j] = 1.0;
        } else {
          top_label[itemid * label_size + j] = 0.0;
        }
      }
    } else { //datum.multi_label_size()>0) { //multiple labels
      for (int j = 0; j < label_size; ++j) {
        top_label[itemid * label_size + j] =
            (datum.multi_label(j));
      }
    }
    */
    // go to the next iter
    layer->iter_->Next();
    if (!layer->iter_->Valid()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      layer->iter_->SeekToFirst();
    }
  }

  return (void*)NULL;
}


template <typename Dtype>
void MultiLabelDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "Data Layer takes two blobs as output.";
  // Initialize the leveldb
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  LOG(INFO) << "Opening leveldb " << this->layer_param_.source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
      << this->layer_param_.source() << std::endl << status.ToString();
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.rand_skip()) {
    unsigned int skip = rand() % this->layer_param_.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      iter_->Next();
      if (!iter_->Valid()) {
        iter_->SeekToFirst();
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(iter_->value().ToString());
  // image
  int cropsize = this->layer_param_.cropsize();
  if (cropsize > 0) {
    (*top)[0]->Reshape(
        this->layer_param_.batchsize(), datum.channels(), cropsize, cropsize);
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.batchsize(), datum.channels(), cropsize, cropsize));
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.batchsize(), datum.channels(), datum.height(),
        datum.width());
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.batchsize(), datum.channels(), datum.height(),
        datum.width()));
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();

  //label size: either label == 1, or size of multilabel
  if (datum.multi_label_size()>0 && !datum.has_label()) {
    datum_label_size_ = datum.multi_label_size(); //multiple labels in multi_label
  } else if (datum.has_label() && datum.multi_label_size()==0) {
    LOG(FATAL) << "The provided data does not have multi_label data, only single label data.";
    //could transform, if we know the number of classes.
    //datum_label_size_ = NUMCLASSES;//single label in label
  } else if (datum.has_label() && datum.multi_label_size()>0) {
    LOG(INFO) <<"The data provided both label, and multi_label data, defaulting to multi_label. ";
    datum_label_size_ = datum.multi_label_size();
  } else {
    LOG(FATAL) << "No labels provided. ";
  }
  // label needs to vary in size
  (*top)[1]->Reshape(this->layer_param_.batchsize(), datum_label_size_, 1, 1);
  prefetch_label_.reset(
      new Blob<Dtype>(this->layer_param_.batchsize(), datum_label_size_, 1, 1));

  CHECK_GT(datum_height_, cropsize);
  CHECK_GT(datum_width_, cropsize);
  // check if we want to have mean
  if (this->layer_param_.has_meanfile()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from" << this->layer_param_.meanfile();
    ReadProtoFromBinaryFile(this->layer_param_.meanfile().c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not do so.
  prefetch_data_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CHECK(!pthread_create(&thread_, NULL, MultiLabelDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void MultiLabelDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
      sizeof(Dtype) * prefetch_data_->count());
  memcpy((*top)[1]->mutable_cpu_data(), prefetch_label_->cpu_data(),
      sizeof(Dtype) * prefetch_label_->count());
  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, MultiLabelDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void MultiLabelDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(),
      prefetch_data_->cpu_data(), sizeof(Dtype) * prefetch_data_->count(),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((*top)[1]->mutable_gpu_data(),
      prefetch_label_->cpu_data(), sizeof(Dtype) * prefetch_label_->count(),
      cudaMemcpyHostToDevice));
  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, MultiLabelDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype MultiLabelDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype MultiLabelDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(MultiLabelDataLayer);

}  // namespace caffe
