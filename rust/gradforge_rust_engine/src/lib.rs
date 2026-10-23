use pyo3::prelude::*;
// Импортируем ToPyArray вместо IntoPyArray для работы с новым API
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use numpy::PyArrayMethods;

#[pymodule]
fn gradforge_rust_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfunction]
    fn add<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'_, f64>,
        b: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let array_a = a.as_array();
        let array_b = b.as_array();

        let result = &array_a + &array_b;

        Ok(result.to_pyarray(py))
    }

    #[pyfunction]
    fn sub<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'_, f64>,
        b: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let array_a = a.as_array();
        let array_b = b.as_array();

        let result = &array_a - &array_b;

        Ok(result.to_pyarray(py))
    }

    #[pyfunction]
    fn div<'py>(
        py: Python<'py>, 
        a: PyReadonlyArray2<'_, f64>,
        b: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>>{
        let array_a = a.as_array();
        let array_b = b.as_array();

        let result = &array_a / &array_b; // Было вычитание, исправлено на деление

        Ok(result.to_pyarray(py))
    }

    #[pyfunction]
    fn matmul<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'_, f64>,
        b: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let array_a = a.as_array();
        let array_b = b.as_array();

        let result = array_a.dot(&array_b);

        Ok(result.to_pyarray(py))
    }

    #[pyfunction]
    fn zeros_like<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'_, f64>
    ) -> PyResult<Bound<'py, PyArray2<f64>>>{
        let mut array = a.as_array().to_owned(); 
    
        for elem in array.iter_mut() {
            *elem = 0.0;
        }

        Ok(array.to_pyarray(py))
    }

    #[pyfunction]
    fn ones_like<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'_, f64>
    ) -> PyResult<Bound<'py, PyArray2<f64>>>{
        let mut array = a.as_array().to_owned(); 
    
        for elem in array.iter_mut() {
            *elem = 1.0;
        }
        Ok(array.to_pyarray(py))
    }

    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    
    Ok(())
}
