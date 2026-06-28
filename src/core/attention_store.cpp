/*
 * src/core/attention_store.cpp
 */

 #include "core/attention_store.hpp"

 void AttentionStore::store(
     std::vector<std::vector<float>> matrix,
     std::vector<std::string>        token_labels,
     std::string                     layer_name,
     int                             head_idx
 ) {
     std::lock_guard<std::mutex> lk(mu_);
     matrix_       = std::move(matrix);
     token_labels_ = std::move(token_labels);
     layer_name_   = std::move(layer_name);
     head_idx_     = head_idx;
     has_data_     = true;
 }
 
 AttentionStore::Snapshot AttentionStore::get() const {
     std::lock_guard<std::mutex> lk(mu_);
     return Snapshot{
         matrix_,
         token_labels_,
         layer_name_,
         head_idx_,
         has_data_
     };
 }
 
 void AttentionStore::clear() {
     std::lock_guard<std::mutex> lk(mu_);
     matrix_.clear();
     token_labels_.clear();
     layer_name_.clear();
     head_idx_ = 0;
     has_data_ = false;
 }