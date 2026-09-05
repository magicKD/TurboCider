#define PY_SSIZE_T_CLEAN
#include <Python.h>

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <unistd.h>

using Clock = std::chrono::steady_clock;

typedef struct {
    PyObject_HEAD
    MLModel *model;
    NSArray<NSNumber *> *inputShape;
    NSArray<NSNumber *> *inputStrides;
    NSArray<NSNumber *> *outputShape;
    NSArray<NSNumber *> *outputStrides;
    MLMultiArray *cachedInput;
    MLMultiArray *cachedOutput;
    MLDictionaryFeatureProvider *cachedProvider;
    MLPredictionOptions *cachedOptions;
    void *cachedInputPointer;
    void *cachedOutputPointer;
    int M;
    int K;
    int N;
    int cacheBindings;
    int fastPrediction;
    double compileMs;
    double loadMs;
    unsigned long long inputBindingHits;
    unsigned long long inputBindingMisses;
    unsigned long long outputBindingHits;
    unsigned long long outputBindingMisses;
    int sourceWasCompiled;
} Flux2ANEModel;

static void Flux2ANEModel_dealloc(Flux2ANEModel *self) {
    self->cachedOptions = nil;
    self->cachedProvider = nil;
    self->cachedOutput = nil;
    self->cachedInput = nil;
    self->outputStrides = nil;
    self->outputShape = nil;
    self->inputStrides = nil;
    self->inputShape = nil;
    self->model = nil;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

static PyObject *Flux2ANEModel_new(PyTypeObject *type, PyObject *, PyObject *) {
    auto *self = reinterpret_cast<Flux2ANEModel *>(type->tp_alloc(type, 0));
    if (self) {
        self->model = nil;
        self->M = self->K = self->N = 0;
        self->cacheBindings = 1;
        self->fastPrediction = 0;
    }
    return reinterpret_cast<PyObject *>(self);
}

static int Flux2ANEModel_init(Flux2ANEModel *self, PyObject *args,
                              PyObject *kwargs) {
    const char *path = nullptr;
    const char *functionName = nullptr;
    static const char *names[] = {
        "model_path", "function_name", "m", "k", "n", "cache_bindings",
        "fast_prediction", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "ssiii|pp", const_cast<char **>(names), &path,
            &functionName, &self->M, &self->K, &self->N,
            &self->cacheBindings, &self->fastPrediction)) {
        return -1;
    }
    if (self->M <= 0 || self->K <= 0 || self->N <= 0) {
        PyErr_SetString(PyExc_ValueError, "m, k, and n must be positive");
        return -1;
    }
    @autoreleasepool {
        NSURL *source = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        self->sourceWasCompiled = [source.pathExtension isEqualToString:@"mlmodelc"];
        NSError *error = nil;
        NSURL *compiled = source;
        if (!self->sourceWasCompiled) {
            const auto compileStart = Clock::now();
            PyThreadState *threadState = PyEval_SaveThread();
            compiled = [MLModel compileModelAtURL:source error:&error];
            PyEval_RestoreThread(threadState);
            self->compileMs = std::chrono::duration<double, std::milli>(
                                  Clock::now() - compileStart)
                                  .count();
            if (!compiled) {
                PyErr_SetString(PyExc_RuntimeError, error.localizedDescription.UTF8String);
                return -1;
            }
        }

        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        configuration.modelDisplayName = source.lastPathComponent;
        if (@available(macOS 15.0, *)) {
            configuration.functionName = [NSString stringWithUTF8String:functionName];
            MLOptimizationHints *hints = [MLOptimizationHints new];
            hints.reshapeFrequency = MLReshapeFrequencyHintInfrequent;
            if (self->fastPrediction) {
                hints.specializationStrategy = MLSpecializationStrategyFastPrediction;
            }
            configuration.optimizationHints = hints;
        } else {
            PyErr_SetString(PyExc_RuntimeError, "FLUX.2 ANE bridge requires macOS 15+");
            return -1;
        }

        const auto loadStart = Clock::now();
        PyThreadState *threadState = PyEval_SaveThread();
        MLModel *loaded = [MLModel modelWithContentsOfURL:compiled
                                           configuration:configuration
                                                   error:&error];
        PyEval_RestoreThread(threadState);
        self->loadMs = std::chrono::duration<double, std::milli>(
                           Clock::now() - loadStart)
                           .count();
        if (!loaded) {
            PyErr_SetString(PyExc_RuntimeError, error.localizedDescription.UTF8String);
            return -1;
        }
        self->model = loaded;
        self->inputShape = @[@1, @(self->K), @1, @(self->M)];
        self->inputStrides =
            @[@(self->K * self->M), @1, @(self->K * self->M), @(self->K)];
        self->outputShape = @[@1, @(self->N), @1, @(self->M)];
        self->outputStrides =
            @[@(self->M * self->N), @1, @(self->M * self->N), @(self->N)];
    }
    return 0;
}

static bool acquireContiguousBuffer(PyObject *object, Py_buffer *view,
                                    bool writable, size_t bytes,
                                    const char *label) {
    const int flags = PyBUF_FORMAT | PyBUF_STRIDES | (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, view, flags) != 0) return false;
    if (view->itemsize != 2 || !view->format || std::strcmp(view->format, "e") != 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError,
                     "%s must expose FP16 memory (buffer format 'e')", label);
        return false;
    }
    if (!PyBuffer_IsContiguous(view, 'C') || static_cast<size_t>(view->len) < bytes) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_ValueError,
                     "%s must expose at least %zu bytes of contiguous memory", label, bytes);
        return false;
    }
    return true;
}

static PyObject *Flux2ANEModel_predict(Flux2ANEModel *self, PyObject *args,
                                       PyObject *kwargs) {
    const auto totalStart = Clock::now();
    PyObject *inputObject = nullptr;
    PyObject *outputObject = nullptr;
    static const char *names[] = {"input", "output", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO", const_cast<char **>(names), &inputObject,
            &outputObject)) {
        return nullptr;
    }
    const size_t inputBytes = static_cast<size_t>(self->M) * self->K * sizeof(uint16_t);
    const size_t outputBytes = static_cast<size_t>(self->M) * self->N * sizeof(uint16_t);
    Py_buffer inputView{};
    Py_buffer outputView{};
    if (!acquireContiguousBuffer(inputObject, &inputView, false, inputBytes, "input")) {
        return nullptr;
    }
    if (!acquireContiguousBuffer(outputObject, &outputView, true, outputBytes, "output")) {
        PyBuffer_Release(&inputView);
        return nullptr;
    }

    PyObject *resultObject = nullptr;
    @autoreleasepool {
        const auto setupStart = Clock::now();
        NSError *error = nil;
        const bool inputCacheHit = self->cacheBindings &&
            self->cachedInputPointer == inputView.buf && self->cachedInput != nil &&
            self->cachedProvider != nil;
        MLMultiArray *input = inputCacheHit ? self->cachedInput :
            [[MLMultiArray alloc] initWithDataPointer:inputView.buf
                                               shape:self->inputShape
                                            dataType:MLMultiArrayDataTypeFloat16
                                             strides:self->inputStrides
                                         deallocator:^(void *) {}
                                               error:&error];
        MLDictionaryFeatureProvider *provider = inputCacheHit ? self->cachedProvider :
            (input ? [[MLDictionaryFeatureProvider alloc]
                         initWithDictionary:@{@"x": [MLFeatureValue featureValueWithMultiArray:input]}
                                      error:&error]
                   : nil);
        if (inputCacheHit) {
            ++self->inputBindingHits;
        } else {
            ++self->inputBindingMisses;
            if (self->cacheBindings && input && provider) {
                self->cachedInputPointer = inputView.buf;
                self->cachedInput = input;
                self->cachedProvider = provider;
            }
        }
        if (!input) {
            PyErr_SetString(PyExc_RuntimeError, error.localizedDescription.UTF8String);
        } else {
            const bool outputCacheHit = self->cacheBindings &&
                self->cachedOutputPointer == outputView.buf && self->cachedOutput != nil &&
                self->cachedOptions != nil;
            MLMultiArray *output = outputCacheHit ? self->cachedOutput :
                [[MLMultiArray alloc] initWithDataPointer:outputView.buf
                                                   shape:self->outputShape
                                                dataType:MLMultiArrayDataTypeFloat16
                                                 strides:self->outputStrides
                                             deallocator:^(void *) {}
                                                   error:&error];
            MLPredictionOptions *options = outputCacheHit ? self->cachedOptions :
                                                           [MLPredictionOptions new];
            if (!outputCacheHit && output) {
                options.outputBackings = @{@"y": output};
                ++self->outputBindingMisses;
                if (self->cacheBindings) {
                    self->cachedOutputPointer = outputView.buf;
                    self->cachedOutput = output;
                    self->cachedOptions = options;
                }
            } else if (outputCacheHit) {
                ++self->outputBindingHits;
            }
            if (!output || !provider) {
                PyErr_SetString(PyExc_RuntimeError, error.localizedDescription.UTF8String);
            } else {
                const double setupMs = std::chrono::duration<double, std::milli>(
                                           Clock::now() - setupStart)
                                           .count();
                const auto start = Clock::now();
                id<MLFeatureProvider> prediction =
                    [self->model predictionFromFeatures:provider options:options error:&error];
                const double elapsed = std::chrono::duration<double, std::milli>(
                                           Clock::now() - start)
                                           .count();
                if (!prediction) {
                    PyErr_SetString(PyExc_RuntimeError, error.localizedDescription.UTF8String);
                } else {
                    MLMultiArray *returned =
                        [prediction featureValueForName:@"y"].multiArrayValue;
                    const double totalMs = std::chrono::duration<double, std::milli>(
                                               Clock::now() - totalStart)
                                               .count();
                    const unsigned long long pageSize =
                        static_cast<unsigned long long>(getpagesize());
                    resultObject = Py_BuildValue(
                        "{s:d,s:d,s:d,s:O,s:O,s:O,s:K,s:K,s:K,s:K,s:K,s:K}",
                        "elapsed_ms", elapsed,
                        "setup_ms", setupMs,
                        "total_ms", totalMs,
                        "output_backing_used", returned == output ? Py_True : Py_False,
                        "input_binding_cache_hit", inputCacheHit ? Py_True : Py_False,
                        "output_binding_cache_hit", outputCacheHit ? Py_True : Py_False,
                        "input_binding_hits", self->inputBindingHits,
                        "input_binding_misses", self->inputBindingMisses,
                        "output_binding_hits", self->outputBindingHits,
                        "output_binding_misses", self->outputBindingMisses,
                        "input_page_offset",
                        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(inputView.buf)) % pageSize,
                        "output_page_offset",
                        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(outputView.buf)) % pageSize);
                }
            }
        }
    }
    PyBuffer_Release(&outputView);
    PyBuffer_Release(&inputView);
    return resultObject;
}

static PyObject *Flux2ANEModel_load_metrics(Flux2ANEModel *self, PyObject *) {
    return Py_BuildValue(
        "{s:d,s:d,s:O,s:O,s:K,s:K,s:K,s:K}",
        "compile_ms", self->compileMs,
        "load_ms", self->loadMs,
        "source_was_compiled", self->sourceWasCompiled ? Py_True : Py_False,
        "fast_prediction", self->fastPrediction ? Py_True : Py_False,
        "input_binding_hits", self->inputBindingHits,
        "input_binding_misses", self->inputBindingMisses,
        "output_binding_hits", self->outputBindingHits,
        "output_binding_misses", self->outputBindingMisses);
}

static PyMethodDef Flux2ANEModel_methods[] = {
    {"predict", reinterpret_cast<PyCFunction>(Flux2ANEModel_predict),
     METH_VARARGS | METH_KEYWORDS,
     "Run Core ML synchronously into a caller-owned FP16 output buffer."},
    {"load_metrics", reinterpret_cast<PyCFunction>(Flux2ANEModel_load_metrics),
     METH_NOARGS, "Return compile, load, and binding-cache counters."},
    {nullptr, nullptr, 0, nullptr},
};

static PyTypeObject Flux2ANEModelType = {PyVarObject_HEAD_INIT(nullptr, 0)};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_flux2_ane_bridge",
    "Zero-copy Core ML bridge for MLX-compatible Python buffers.",
    -1,
    nullptr,
};

PyMODINIT_FUNC PyInit__flux2_ane_bridge(void) {
    Flux2ANEModelType.tp_name = "_flux2_ane_bridge.Flux2ANEModel";
    Flux2ANEModelType.tp_basicsize = sizeof(Flux2ANEModel);
    Flux2ANEModelType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    Flux2ANEModelType.tp_doc = "Fixed-shape Core ML model backed by MLX buffers.";
    Flux2ANEModelType.tp_new = Flux2ANEModel_new;
    Flux2ANEModelType.tp_init = reinterpret_cast<initproc>(Flux2ANEModel_init);
    Flux2ANEModelType.tp_dealloc = reinterpret_cast<destructor>(Flux2ANEModel_dealloc);
    Flux2ANEModelType.tp_methods = Flux2ANEModel_methods;
    if (PyType_Ready(&Flux2ANEModelType) < 0) return nullptr;
    PyObject *result = PyModule_Create(&module);
    if (!result) return nullptr;
    Py_INCREF(&Flux2ANEModelType);
    if (PyModule_AddObject(result, "Flux2ANEModel",
                           reinterpret_cast<PyObject *>(&Flux2ANEModelType)) < 0) {
        Py_DECREF(&Flux2ANEModelType);
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}

