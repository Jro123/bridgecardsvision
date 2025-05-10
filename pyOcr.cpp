#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cstdlib> 
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

bool initie = false;

void initialize_python_environment() {
    PyConfig config;
    // Définir les variables d'environnement PYTHONHOME et PYTHONPATH 
    #ifdef _WIN32
    // Windows specific environment settings
    //_putenv_s("PYTHONHOME", "C:\\Program Files\\Python312");
    //_putenv_s("PYTHONPATH", "C:\\Program Files\\Python312\\Lib;C:\\Program Files\\Python312\\DLLs;C:\\Program Files\\Python312\\Lib\\site-packages");
    std::cout << "PYTHONHOME: " << getenv("PYTHONHOME") << std::endl;
    std::cout << "PYTHONPATH: " << getenv("PYTHONPATH") << std::endl;

    PyStatus status;

    PyConfig_InitPythonConfig(&config);
    status = PyConfig_SetBytesString(&config, &config.program_name, "D:\\DEV\\CV2C\\X64\\Debug\\CV2C.exe");
    if (PyStatus_Exception(status)) {
        goto exception;
    }
    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) {
        goto exception;
    }
    PyConfig_Clear(&config);
    #else
    // Linux specific environment settings
    //setenv("PYTHONHOME", "/usr/local/python3.12", 1);
    //setenv("PYTHONPATH", "/usr/local/python3.12/lib:/usr/local/python3.12/lib/site-packages", 1);
    std::cout << "PYTHONHOME: " << getenv("PYTHONHOME") << std::endl;
    std::cout << "PYTHONPATH: " << getenv("PYTHONPATH") << std::endl;
    Py_Initialize();
    #endif

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("print(sys.path)");
    std::cout << "sys.path:" << std::endl << std::endl;
    
    #ifdef _WIN32
    PyRun_SimpleString("sys.path = ['', 'C:\\\\Program Files\\\\Python312\\\\Scripts', 'C:\\\\Program Files\\\\Python312\\\\DLL', 'C:\\\\Program Files\\\\Python312\\\\LIB', 'C:\\\\Program Files\\\\Python312\\\\site-packages', 'C:\\\\Program Files\\\\Python312', 'C:\\\\Program Files\\\\Python312\\\\python312_d.zip',  'C:\\\\Program Files\\\\Python312\\\\python312.zip', 'C:\\\\Program Files\\\\Python312\\\\DLLs',  'C:\\\\Program Files\\\\Python312\\\\Lib\\\\site-packages']");
    PyRun_SimpleString("print(sys.path)");
    std::cout << std::endl;
    #endif

    return;

    exception:
        PyConfig_Clear(&config);
        return;
}

void chargermodele(PyObject* pModule) {
    const char * function_name = "chargerModele";
    // Charger la fonction Python
    PyObject* pFunc = PyObject_GetAttrString(pModule, function_name);
    if (!pFunc || !PyCallable_Check(pFunc)) {
        PyErr_Print();
        std::cerr << "Erreur: Impossible de trouver la fonction " << function_name << std::endl;
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
        return;
    }

    // Préparer les arguments pour la fonction Python
    PyObject* pArgs = PyTuple_Pack(0);
    if (!pArgs) {
        PyErr_Print();
        std::cerr << "Erreur: Impossible de préparer les arguments" << std::endl;
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
        return;
    }

    // Appeler la fonction Python
    PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
    const char* result_text("");
    if (pValue) {
        if (PyUnicode_Check(pValue)) {
            result_text = PyUnicode_AsUTF8(pValue);
            std::cout << "Résultat de la fonction Python : " << result_text << std::endl;
        } else {
            std::cerr << "Erreur: La fonction Python n'a pas retourné une chaîne de caractères." << std::endl; 
        }

        Py_DECREF(pValue);
    } else {
        PyErr_Print();
        std::cerr << "Erreur: Impossible d'appeler la fonction" << std::endl;
    }

    // Nettoyer les références
    Py_DECREF(pArgs);
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
}

PyObject* pModule;
PyObject* pFunc;
bool estOK = true;

#ifdef _WIN32
// Fonction pour convertir std::string en std::wstring 
std::wstring string_to_wstring(const std::string& str) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size_needed);
    return wstr;
}

// Fonction pour activer l'environnement virtuel 
void activate_virtualenv(const std::wstring& venv_path) {
    std::wstring activate_cmd = venv_path + L"\\Scripts\\activate.bat";
    _wsystem(activate_cmd.c_str());
}
#endif

cv::String pyOCR(cv::Mat image, double *pconfiance, double* pangle) {
    if (!estOK) return "";

    PyObject* pName;
    const char* module_name = "execOCR";
    const char* function_name = "process_image";
    
    if (!initie) {
        // Activer l'environnement virtuel avant d'appeler Python 
        #ifdef _WIN32
        _putenv("VIRTUAL_ENV=C:\\Users\\rossi_000\\myenv");
        _putenv("PATH=D:\\;C:\\Users\\rossi_000\\myenv\\Scripts;C:\\Users\\rossi_000\\myenv\\Lib;C:\\Users\\rossi_000\\myenv\\DLLs;c:\\\\python\\Lib\\site-packages");
        #else
        setenv("VIRTUAL_ENV", "/home/jro/myenv", 1);
        setenv("PATH", "/home/jro/myenv/bin:/usr/local/bin:/usr/bin:/bin", 1);
        #endif

        Py_Initialize();
        PyObject* sys_path = PySys_GetObject("path");

        #ifdef _WIN32
        PyList_Append(sys_path, PyUnicode_FromString("D:\\"));
        #else
        PyList_Append(sys_path, PyUnicode_FromString("/home/jro/DEV/CV2C"));
        #endif

        // Charger le module Python 
        pName = PyUnicode_DecodeFSDefault(module_name);
        pModule = PyImport_Import(pName);
        Py_DECREF(pName);
        if (!pModule) {
            PyErr_Print();
            std::cerr << "Erreur: Impossible de charger le module " << module_name << std::endl;
            initie = true;
            estOK = false;
            return "";
        }

        // Charger la fonction Python
        pFunc = PyObject_GetAttrString(pModule, function_name);
        if (!pFunc || !PyCallable_Check(pFunc)) {
            PyErr_Print();
            std::cerr << "Erreur: Impossible de trouver la fonction " << function_name << std::endl;
            Py_XDECREF(pFunc);
            Py_DECREF(pModule);
            return "";
        }

        initie = true;
    }

    // convertir l'image en tableau d'octets
    int tot = image.total();
    int elmsz = image.elemSize();
    std::vector<uchar> image_bytes(image.data, image.data + image.total() * image.elemSize());

    // Préparer les arguments pour la fonction Python
    PyObject* pArgs = PyTuple_Pack(3, PyBytes_FromStringAndSize((char*)image_bytes.data(),
        image_bytes.size()), PyLong_FromLong(image.cols), PyLong_FromLong(image.rows));
    if (!pArgs) {
        PyErr_Print();
        std::cerr << "Erreur: Impossible de préparer les arguments" << std::endl;
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
        return "";
    }

    // Appeler la fonction Python
    PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
    Py_DECREF(pArgs);
    const char* result_text("");
    if (pValue) {
        std::string result = PyUnicode_AsUTF8(pValue);
        Py_DECREF(pValue);
        std::cout << "Résultat de la fonction Python : " << result << std::endl;
        
        // Séparer l'indice de confiance et le texte
        std::stringstream ss(result);
        std::string confidence_str;
        std::string text;
        std::string orientation; 
        std::getline(ss, confidence_str, ',');
        std::getline(ss, text, ',');
        std::getline(ss,orientation);

        double confidence = std::stod(confidence_str);
        double angle = std::stod(orientation);
        if(pconfiance) *pconfiance = confidence;
        if(pangle) *pangle = angle;
        return text;
    } else {
        PyErr_Print();
        std::cerr << "Erreur: Appeler la fonction " << function_name << " a échoué" << std::endl;
         Py_XDECREF(pFunc);
          Py_DECREF(pModule);
           return "";
         } 
         // Nettoyer les références Py_DECREF(pArgs);
          //Py_XDECREF(pFunc);
           //Py_DECREF(pModule);
            // Finaliser l'interpréteur Python //Py_Finalize();
             cv::String cvstr(result_text); return cvstr;
             }
