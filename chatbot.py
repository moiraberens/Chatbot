import gzip
import pickle
import seq2seq_model_adapted
import tensorflow as tf
import random


## ------------------ ##
## Settings for model ##
## ------------------ ##

buckets = [(5,7), (10,12), (15,17), (20,22)]

settings = {}
settings['vocab_size'] = 0
settings['buckets'] = len(buckets)
settings['num_units_per_layer'] = 2048
settings['num_layers'] = 2
settings['max_gredient_norm'] = 5.0
settings['batch_size'] = 64
settings['learning_rate'] = 0.5
settings['learning_rate_decay'] = 0.99
settings['num_samples'] = 1024
settings['train_dir'] = '\tmp'

## ------------------ ##
##     functions      ##
## ------------------ ##

def load(path):
    with gzip.open(path,"r") as f:
        item = pickle.load(f)
    return item


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
        settings['vocab_size'],
        settings['vocab_size'],
        buckets,
        settings['num_units_per_layer'],
        settings['num_layer'],
        settings['max_gredient_norm'],
        settings['batch_size'],
        settings['learning_rate'],
        settings['learning_rate_decay'],
        use_lstm=False)
    #this code is usefull if we want a checkpoint I ignore this for now
    #ckpt = tf.train.get_checkpoint_state(settings['train_dir'])
    #if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        #  print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        #  model.saver.restore(session, ckpt.model_checkpoint_path)
    #else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    return model







    
    

## ------------------ ##
##        Main        ##
## ------------------ ##

if __name__ == "__main__":
    # Load Dataset
    dataset = load("Data/dataset.pkl.gz")
    vocab_text2token = load("Data/vocab_text2token.pkl.gz")
    settings['vocab_size'] = len(vocab_text2token) 
    bucket_id = 0
    
    #test chatbot
    #"""Test the translation model."""
    #with tf.Session() as sess:
    #    print("Self-test for neural translation model.")
    #    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    #    model = seq2seq_model_adapted.Seq2SeqModel(settings['vocab_size'], settings['vocab_size'], buckets, 32, 1,
    #                                   5.0, 32, 0.3, 0.99)
    #    sess.run(tf.global_variables_initializer())
    #    
    #    for _ in range(5):  # Train the fake model for 5 steps.
    #        bucket_id = random.choice([0, 1])
    #        encoder_inputs, decoder_inputs, target_weights = model.get_batch(settings, dataset, buckets, bucket_id)
    #        model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
    
    #train chatbot
    #"""Test the translation model."""
    """Train a en->fr translation model using WMT data."""
    from_train = None
    to_train = None
    from_dev = None
    to_dev = None
    if FLAGS.from_train_data and FLAGS.to_train_data:
        from_train_data = FLAGS.from_train_data
        to_train_data = FLAGS.to_train_data
        from_dev_data = from_train_data
        to_dev_data = to_train_data
        if FLAGS.from_dev_data and FLAGS.to_dev_data:
            from_dev_data = FLAGS.from_dev_data
            to_dev_data = FLAGS.to_dev_data
        from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
            FLAGS.data_dir,
            from_train_data,
            to_train_data,
            from_dev_data,
            to_dev_data,
            FLAGS.from_vocab_size,
            FLAGS.to_vocab_size)
        else:
          # Prepare WMT data.
          print("Preparing WMT data in %s" % FLAGS.data_dir)
          from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_wmt_data(
              FLAGS.data_dir, FLAGS.from_vocab_size, FLAGS.to_vocab_size)
    
      with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)
    
        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)."
               % FLAGS.max_train_data_size)
        dev_set = read_data(from_dev, to_dev)
        train_set = read_data(from_train, to_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
    
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
    
        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
          # Choose a bucket according to data distribution. We pick a random number
          # in [0, 1] and use the corresponding interval in train_buckets_scale.
          random_number_01 = np.random.random_sample()
          bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])
    
          # Get a batch and make a step.
          start_time = time.time()
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              train_set, bucket_id)
          _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, False)
          step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
          loss += step_loss / FLAGS.steps_per_checkpoint
          current_step += 1
    
          # Once in a while, we save checkpoint, print statistics, and run evals.
          if current_step % FLAGS.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
            print ("global step %d learning rate %.4f step-time %.2f perplexity "
                   "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                             step_time, perplexity))
            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0, 0.0
            # Run evals on development set and print their perplexity.
            for bucket_id in xrange(len(_buckets)):
              if len(dev_set[bucket_id]) == 0:
                print("  eval: empty bucket %d" % (bucket_id))
                continue
              encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                  dev_set, bucket_id)
              _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)
              eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                  "inf")
              print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
            sys.stdout.flush()
    