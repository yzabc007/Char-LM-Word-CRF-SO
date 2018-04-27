import os
import tensorflow as tf


class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, params):
        self.params = params
        self.sess = None
        self.saver = None

    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        # self.logger.info("Initializing tf session")
        print("Initializing tf session")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def restore_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.params['save_model_path']):
            os.makedirs(self.params['save_model_path'])
        self.saver.save(self.sess, self.params['save_model_path'])

    def close_session(self):
        """Closes the session"""
        self.sess.close()

    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.params['save_predict_path'], self.sess.graph)

    def train(self, train, dev, test):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        best_score = 0
        nepoch_no_imprv = 0 # for early stopping
        self.add_summary() # tensorboard

        for epoch in range(self.params['train_epochs']):
            print("Epoch {:} out of {:}".format(epoch + 1, self.params['train_epochs']))
            # self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))

            score = self.run_epoch(train, dev, epoch)
            self.params['lr_rate'] *= self.params['weight_decay']
            # self.config.lr *= self.config.lr_decay # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                print('Evaluate on test set: ')
                self.run_evaluate(test)
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                # self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.params['patiences']:
                    print("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                    # self.logger.info("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                    break

    def evaluate(self, test):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)