/*
Library from https://github.com/datagraph/jquery-jsonrpc

Authors
Josh Huckabee joshhuckabee@gmail.com
Chris Petersen cpetersen@mythtv.org
Jerry Ablan opensource@pogostick.com

'License'

This plugin is free and unemcumbered software released into the public domain.
* 
* Adapted to msgpack by Brian Maranville
* 
* This work is in the public domain
* 
*/

// require(msgpack);

(function($, undefined) {
  $.extend({
    msgpackRPC: {
      // RPC Version Number
      version: '5',

      // End point URL, sets default in requests if not
      // specified with the request call
      endPoint: null,

      // Default namespace for methods
      namespace: null,

      /*
       * Provides the RPC client with an optional default endpoint and namespace
       *
       * @param {object} The params object which can contain
       *   endPoint {string} The default endpoint for RPC requests
       *   namespace {string} The default namespace for RPC requests
       *   cache {boolean} If set to false, it will force requested
       *       pages not to be cached by the browser. Setting cache
       *       to false also appends a query string parameter,
       *       "_=[TIMESTAMP]", to the URL. (Default: true)
       */
      setup: function(params) {
        this._validateConfigParams(params);
        this.endPoint = params.endPoint;
        this.namespace = params.namespace;
        this.cache = params.cache !== undefined ? params.cache : true;
        return this;
      },

      /*
       * Convenience wrapper method to allow you to temporarily set a config parameter
       * (endPoint or namespace) and ensure it gets set back to what it was before
       *
       * @param {object} The params object which can contains
       *   endPoint {string} The default endpoint for RPC requests
       *   namespace {string} The default namespace for RPC requests
       * @param {function} callback The function to call with the new params in place
       */
      withOptions: function(params, callback) {
        this._validateConfigParams(params);
        // No point in running if there isn't a callback received to run
        if(callback === undefined) throw("No callback specified");

        origParams = {endPoint: this.endPoint, namespace: this.namespace};
        this.setup(params);
        callback.call(this);
        this.setup(origParams);
      },

      /*
       * Performas a single RPC request
       *
       * @param {string} method The name of the rpc method to be called
       * @param {object} options A collection of object which can contains
       *  params {array} the params array to send along with the request
       *  success {function} a function that will be executed if the request succeeds
       *  error {function} a function that will be executed if the request fails
       *  url {string} the url to send the request to
       *  id {string} the provenance id for this request (defaults to 1)
       *  cache {boolean} If set to false, it will force requested
       *       pages not to be cached by the browser. Setting cache
       *       to false also appends a query string parameter,
       *       "_=[TIMESTAMP]", to the URL. (Default: cache value
       *       set with the setup method)
       * @return {undefined}
       */
      request: function(method, options) {
        if(options === undefined) {
          options = { id: 1 };
        }
        if (options.id === undefined) {
          options.id = 1;
        }
        if (options.cache === undefined) {
          options.cache = this.cache;
        }

        // Validate method arguments
        this._validateRequestMethod(method);
        this._validateRequestParams(options.params);
        this._validateRequestCallbacks(options.success, options.error);

        // Perform the actual request
        this._doRequest(msgpack.encode(this._requestDataObj(method, options.params, options.id)), options);

        return true;
      },


      // Validate a params hash
      _validateConfigParams: function(params) {
        if(params === undefined) {
          throw("No params specified");
        }
        else {
          if(params.endPoint && typeof(params.endPoint) !== 'string'){
            throw("endPoint must be a string");
          }
          if(params.namespace && typeof(params.namespace) !== 'string'){
            throw("namespace must be a string");
          }
        }
      },

      // Request method must be a string
      _validateRequestMethod: function(method) {
        if(typeof(method) !== 'string') throw("Invalid method supplied for msgpackRPC request")
        return true;
      },

      // Validate request params.  Must be a) empty, b) an object (e.g. {}), or c) an array
      _validateRequestParams: function(params) {
        if(!($.isArray(params))) {
          throw("Invalid params supplied for msgpackRPC request. It must be an array.");
        }
        return true;
      },

      _validateRequestCallbacks: function(success, error) {
        // Make sure callbacks are either empty or a function
        if(success !== undefined &&
           typeof(success) !== 'function') throw("Invalid success callback supplied for msgpackRPC request");
        if(error !== undefined &&
         typeof(error) !== 'function') throw("Invalid error callback supplied for msgpackRPC request");
        return true;
      },

      // Internal method used for generic ajax requests
      _doRequest: function(data, options) {
        var _that = this;
        $.ajax({
          type: 'POST',
          async: false !== options.async,
          dataType: "binary",
          responseType: "arraybuffer",
          url: this._requestUrl((options.endPoint || options.url), options.cache),
          data: data,
          cache: options.cache,
          processData: false,
          error: function(msg) {
            _that._requestError.call(_that, msg, options.error);
          },
          success: function(msg) {
            _that._requestSuccess.call(_that, msg, options.success, options.error);
          }
        })
      },

      // Determines the appropriate request URL to call for a request
      _requestUrl: function(url, cache) {
        url = url || this.endPoint;
        if (!cache) {
            if (url.indexOf("?") < 0) {
              url += '?tm=' + new Date().getTime();
            }
            else {
              url += "&tm=" + new Date().getTime();
            }
        }
        return url;
      },

      // Creates an RPC suitable request object
      // as per the spec at: https://github.com/msgpack-rpc/msgpack-rpc/blob/master/spec.md
      _requestDataObj: function(method, params, id) {
        var dataObj = [
          0, // identifies "Request" message
          id, // msgid: should be 32-bit unsigned integer
          this.namespace ? this.namespace +'.'+ method : method, // method string
          params || [] // params list
        ]
        return dataObj;
      },

      // Handles calling of error callback function
      _requestError: function(msg, error) {
        if (error !== undefined && typeof(error) === 'function') {
          //msg = msgpack.decode(msg);
          if(typeof(msg.responseText) === 'string') {
            try {
              error(eval ( '(' + msg.responseText + ')' ));
            }
            catch(e) {
              error(this._response());
            }
          }
          else {
            error(this._response());
          }
        }
      },

      // Handles calling of RPC success, calls error callback
      // if the response contains an error
      // TODO: Handle error checking for batch requests
      _requestSuccess: function(msg, success, error) {
        var response = this._response(msg);

        // If we've encountered an error in the response, trigger the error callback if it exists
        if(response[2] != null && typeof(error) === 'function') {
          error(response);
          return;
        }

        // Otherwise, successful request, run the success request if it exists
        if(typeof(success) === 'function') {
          success(response[3]);
        }
      },

      // Returns a generic msgpack RPC 5 compatible response object
      _response: function(msg) {
        if (msg === undefined) {
          return {
            error: 'Internal server error',
            version: '5'
          };
        }
        else {
          try {
            if(msg instanceof ArrayBuffer) {
              var responseArray = new Uint8Array(msg); 
              msg = msgpack.decode(responseArray);
            }

            if (!($.isArray(msg) && msg.length == 4)) {
              throw 'Response error: must be array of length 4';
            }

            return msg;
          }
          catch (e) {
            return {
              error: 'Internal server error: ' + e,
              version: '5'
            }
          }
        }
      }

    }
  });
})(jQuery);
