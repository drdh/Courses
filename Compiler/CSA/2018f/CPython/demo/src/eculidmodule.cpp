#include <Python.h>
#include <new>
#include "eculid.h"

using namespace eculid;

static PyObject *EculidError;

struct eculid_Net_t
{
    PyObject_HEAD Net v;
};

static PyTypeObject eculid_Net_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "eculid.Net",
    sizeof(eculid_Net_t),
};

/*
static void eculid_Net_dealloc(PyObject *self)
{
    ((eculid_Net_t *)self)->v.eculid::Net::~Net();
    PyObject_Del(self);
}
*/

static PyObject *
eculid_init(PyObject *self, PyObject *args)
{
    eculid_Net_t *m = PyObject_NEW(eculid_Net_t, &eculid_Net_Type);
    new (&m->v) Net();
    Py_INCREF(m);
    return (PyObject *)m;
}

static PyObject *
eculid_gcd(PyObject *self, PyObject *args)
{
    int m, n;
    int i;

    if (!PyArg_ParseTuple(args, "ii", &m, &n))
        return NULL;
    
    if (m <= 0 || n <= 0) {
        PyErr_SetString(EculidError, "invalid gcd argument");
        return NULL;
    }
    
    i = gcd(m, n);

    return PyLong_FromLong(i);
}

static PyObject *
eculid_Net_forward(PyObject *self, PyObject *args)
{
    eculid::Net *_self_ = NULL;
    if (PyObject_TypeCheck(self, &eculid_Net_Type))
        _self_ = &((eculid_Net_t *)self)->v;
    if (_self_ == NULL)
        PyErr_SetString(EculidError, "Incorrect type of self");

    _self_->forward();

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef EculidMethods[] = {
    {"gcd", eculid_gcd, METH_VARARGS, "greatest common divisor"},
    {"init", eculid_init, METH_VARARGS, "eculid_init"},
    {NULL, NULL, 0, NULL}
};

static PyMethodDef EculidNetMethods[] = {
    {"forward", (PyCFunction)eculid_Net_forward, METH_VARARGS, "eculid_Net_forward"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef eculidmodule = {
    PyModuleDef_HEAD_INIT,
    "eculid",
    "eculid algorithm",
    -1,
    EculidMethods
};

extern "C" PyMODINIT_FUNC
PyInit_eculid(void)
{
    PyObject *m;

    eculid_Net_Type.tp_new = PyType_GenericNew;
    eculid_Net_Type.tp_methods = EculidNetMethods;
    if (PyType_Ready(&eculid_Net_Type) < 0)
        return NULL;

    m = PyModule_Create(&eculidmodule);
    if (m == NULL)
        return NULL;

    EculidError = PyErr_NewException("eculid.error", NULL, NULL);
    Py_INCREF(EculidError);
    PyModule_AddObject(m, "error", EculidError);
    
    Py_INCREF(&eculid_Net_Type);
    PyModule_AddObject(m, "Net", (PyObject *)&eculid_Net_Type);

    return m;
}
