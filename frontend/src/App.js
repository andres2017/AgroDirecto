import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [productos, setProductos] = useState([]);
  const [loading, setLoading] = useState(false);
  const [iaStatus, setIaStatus] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [formData, setFormData] = useState({
    nombre: '',
    cantidad_kg: 1
  });
  const [publishResult, setPublishResult] = useState(null);
  const [activeTab, setActiveTab] = useState('marketplace');

  useEffect(() => {
    fetchProductos();
    fetchIAStatus();
  }, []);

  const fetchProductos = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${BACKEND_URL}/api/productos`);
      setProductos(response.data);
    } catch (error) {
      console.error('Error fetching productos:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchIAStatus = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/status_ia`);
      setIaStatus(response.data);
    } catch (error) {
      console.error('Error fetching IA status:', error);
    }
  };

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => setImagePreview(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handlePublishProduct = async (e) => {
    e.preventDefault();
    
    if (!selectedImage) {
      alert('Por favor selecciona una imagen del producto');
      return;
    }

    try {
      setLoading(true);
      setPublishResult(null);

      const formDataToSend = new FormData();
      formDataToSend.append('imagen', selectedImage);
      formDataToSend.append('nombre', formData.nombre);
      formDataToSend.append('cantidad_kg', formData.cantidad_kg);

      const response = await axios.post(
        `${BACKEND_URL}/api/publicar_producto`,
        formDataToSend,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      setPublishResult(response.data);
      
      // Reset form
      setSelectedImage(null);
      setImagePreview(null);
      setFormData({ nombre: '', cantidad_kg: 1 });
      
      // Refresh products
      fetchProductos();
      
    } catch (error) {
      console.error('Error publishing product:', error);
      setPublishResult({
        estado: 'error',
        mensaje: error.response?.data?.detail || 'Error al publicar producto'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleComprarProducto = async (producto) => {
    const cantidad = prompt(`¿Cuántos kg de ${producto.nombre} deseas comprar? (Disponible: ${producto.cantidad_kg}kg)`);
    
    if (!cantidad || cantidad <= 0) return;
    
    if (parseFloat(cantidad) > producto.cantidad_kg) {
      alert('Cantidad solicitada excede el stock disponible');
      return;
    }

    const metodoEntrega = window.confirm('¿Deseas entrega a domicilio? (Cancelar para recogida en cultivo)') 
      ? 'domicilio' : 'recogida_cultivo';

    try {
      setLoading(true);
      
      const compraData = {
        producto_id: producto.id,
        cantidad_kg: parseFloat(cantidad),
        metodo_entrega: metodoEntrega,
        comprador_info: {
          nombre: 'Cliente Demo',
          contacto: 'demo@agrodirecto.com'
        }
      };

      const response = await axios.post(`${BACKEND_URL}/api/comprar_producto`, compraData);
      
      alert(`¡Compra realizada exitosamente! Total: $${response.data.total_cop} COP`);
      fetchProductos(); // Refresh products
      
    } catch (error) {
      console.error('Error purchasing product:', error);
      alert('Error al realizar la compra: ' + (error.response?.data?.detail || 'Error desconocido'));
    } finally {
      setLoading(false);
    }
  };

  const handleTrainModels = async () => {
    if (!window.confirm('El entrenamiento de modelos puede tomar varios minutos. ¿Continuar?')) {
      return;
    }

    try {
      setLoading(true);
      const response = await axios.post(`${BACKEND_URL}/api/entrenar_modelos`);
      alert(response.data.mensaje);
      fetchIAStatus(); // Refresh IA status
    } catch (error) {
      console.error('Error training models:', error);
      alert('Error entrenando modelos: ' + (error.response?.data?.detail || 'Error desconocido'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b-2 border-green-200">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-green-600 rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-lg">🌱</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-800">AgroDirecto Tunja</h1>
                <p className="text-sm text-gray-600">Agricultura Inteligente con IA</p>
              </div>
            </div>
            
            {/* IA Status */}
            <div className="flex items-center space-x-4">
              {iaStatus && (
                <div className="bg-gray-100 rounded-lg px-3 py-2 text-sm">
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${iaStatus.cnn_activa ? 'bg-green-500' : 'bg-red-500'}`}></div>
                    <span>CNN: {iaStatus.cnn_activa ? 'Activa' : 'Inactiva'}</span>
                  </div>
                  <div className="flex items-center space-x-2 mt-1">
                    <div className={`w-2 h-2 rounded-full ${iaStatus.dnn_activa ? 'bg-green-500' : 'bg-red-500'}`}></div>
                    <span>DNN: {iaStatus.dnn_activa ? 'Activa' : 'Inactiva'}</span>
                  </div>
                </div>
              )}
              
              {iaStatus && !iaStatus.modelos_entrenados && (
                <button
                  onClick={handleTrainModels}
                  disabled={loading}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
                >
                  Entrenar Modelos IA
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b">
        <div className="container mx-auto px-6">
          <div className="flex space-x-8">
            <button
              onClick={() => setActiveTab('marketplace')}
              className={`py-4 px-2 border-b-2 font-medium text-sm ${
                activeTab === 'marketplace' 
                  ? 'border-green-500 text-green-600' 
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              🛒 Marketplace
            </button>
            <button
              onClick={() => setActiveTab('publicar')}
              className={`py-4 px-2 border-b-2 font-medium text-sm ${
                activeTab === 'publicar' 
                  ? 'border-green-500 text-green-600' 
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              📸 Publicar Producto
            </button>
          </div>
        </div>
      </nav>

      <div className="container mx-auto px-6 py-8">
        {/* Marketplace Tab */}
        {activeTab === 'marketplace' && (
          <div>
            <div className="mb-6">
              <h2 className="text-3xl font-bold text-gray-800 mb-2">Productos Disponibles</h2>
              <p className="text-gray-600">Productos frescos directamente de las fincas de Tunja</p>
            </div>

            {loading && (
              <div className="flex justify-center items-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500"></div>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {productos.map((producto) => (
                <div key={producto.id} className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow">
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-semibold text-gray-800 mb-1">{producto.nombre}</h3>
                        <p className="text-sm text-gray-500 capitalize">{producto.tipo_producto}</p>
                      </div>
                      <div className="text-right">
                        <div className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${
                          producto.calidad_ia === 'Excelente' ? 'bg-green-100 text-green-800' :
                          producto.calidad_ia === 'Buena' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {producto.calidad_ia}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-3 mb-4">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-600">Precio por kg:</span>
                        <span className="text-2xl font-bold text-green-600">
                          ${producto.precio_sugerido_cop.toLocaleString()} COP
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <span className="text-gray-600">Disponible:</span>
                        <span className="font-medium">{producto.cantidad_kg} kg</span>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-gray-600">Confianza IA:</span>
                        <span className="text-sm text-blue-600">{producto.confianza_calidad_pct}%</span>
                      </div>
                    </div>

                    <div className="border-t pt-4 mb-4">
                      <div className="text-sm text-gray-600">
                        <p><strong>Finca:</strong> {producto.origen_finca}</p>
                        <p><strong>Agricultor:</strong> {producto.agricultor}</p>
                        <p><strong>Blockchain ID:</strong> {producto.blockchain_id}</p>
                      </div>
                    </div>

                    <button
                      onClick={() => handleComprarProducto(producto)}
                      disabled={loading || producto.cantidad_kg <= 0}
                      className="w-full bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white font-medium py-3 px-4 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {producto.cantidad_kg > 0 ? 'Comprar Producto' : 'Agotado'}
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {productos.length === 0 && !loading && (
              <div className="text-center py-12">
                <div className="text-6xl mb-4">🌾</div>
                <h3 className="text-xl text-gray-600 mb-2">No hay productos disponibles</h3>
                <p className="text-gray-500">¡Sé el primero en publicar un producto!</p>
              </div>
            )}
          </div>
        )}

        {/* Publicar Producto Tab */}
        {activeTab === 'publicar' && (
          <div className="max-w-2xl mx-auto">
            <div className="mb-6">
              <h2 className="text-3xl font-bold text-gray-800 mb-2">Publicar Producto</h2>
              <p className="text-gray-600">Usa IA para analizar la calidad y obtener precio justo</p>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-8">
              <form onSubmit={handlePublishProduct} className="space-y-6">
                {/* Image Upload */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Imagen del Producto (Análisis con IA)
                  </label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-green-400 transition-colors">
                    {imagePreview ? (
                      <div className="space-y-4">
                        <img
                          src={imagePreview}
                          alt="Preview"
                          className="mx-auto h-48 w-48 object-cover rounded-lg"
                        />
                        <button
                          type="button"
                          onClick={() => {
                            setSelectedImage(null);
                            setImagePreview(null);
                          }}
                          className="text-red-600 hover:text-red-700 text-sm"
                        >
                          Cambiar imagen
                        </button>
                      </div>
                    ) : (
                      <div>
                        <div className="text-4xl mb-2">📸</div>
                        <p className="text-gray-600 mb-2">Selecciona una imagen de tu producto</p>
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleImageSelect}
                          className="hidden"
                          id="image-upload"
                        />
                        <label
                          htmlFor="image-upload"
                          className="cursor-pointer bg-green-50 hover:bg-green-100 text-green-700 px-4 py-2 rounded-lg font-medium transition-colors"
                        >
                          Subir Imagen
                        </label>
                      </div>
                    )}
                  </div>
                </div>

                {/* Product Details */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Nombre del Producto (Opcional)
                    </label>
                    <input
                      type="text"
                      name="nombre"
                      value={formData.nombre}
                      onChange={handleInputChange}
                      placeholder="La IA sugerirá un nombre"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Cantidad (kg)
                    </label>
                    <input
                      type="number"
                      name="cantidad_kg"
                      value={formData.cantidad_kg}
                      onChange={handleInputChange}
                      min="0.1"
                      step="0.1"
                      required
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading || !selectedImage}
                  className="w-full bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white font-medium py-3 px-4 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <div className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                      Analizando con IA...
                    </div>
                  ) : (
                    '🤖 Analizar y Publicar con IA'
                  )}
                </button>
              </form>

              {/* Publish Result */}
              {publishResult && (
                <div className={`mt-6 p-4 rounded-lg ${
                  publishResult.estado === 'exito' ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
                }`}>
                  <div className={`font-medium ${
                    publishResult.estado === 'exito' ? 'text-green-800' : 'text-red-800'
                  }`}>
                    {publishResult.mensaje}
                  </div>
                  
                  {publishResult.estado === 'exito' && publishResult.producto && (
                    <div className="mt-3 text-sm text-green-700">
                      <div className="grid grid-cols-2 gap-2">
                        <div><strong>Tipo:</strong> {publishResult.producto.tipo_producto}</div>
                        <div><strong>Calidad:</strong> {publishResult.producto.calidad_ia}</div>
                        <div><strong>Precio sugerido:</strong> ${publishResult.producto.precio_sugerido_cop} COP/kg</div>
                        <div><strong>Confianza:</strong> {publishResult.producto.confianza_calidad_pct}%</div>
                      </div>
                      <div className="mt-2">
                        <strong>Blockchain ID:</strong> {publishResult.producto.blockchain_id}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-12">
        <div className="container mx-auto px-6 text-center">
          <div className="mb-4">
            <h3 className="text-lg font-semibold mb-2">AgroDirecto Tunja</h3>
            <p className="text-gray-300">Conectando agricultores y consumidores con tecnología IA</p>
          </div>
          <div className="text-sm text-gray-400">
            <p>🤖 Modelos de IA: CNN para Calidad + DNN para Precios</p>
            <p>🔗 Blockchain para Trazabilidad</p>
            <p>🌱 Agricultura Sostenible en Boyacá</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;