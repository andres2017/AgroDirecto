from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timedelta
import asyncio
import io
import random
import math

# AI/ML imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
from PIL import Image

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="AgroDirecto Tunja API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AI Models directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Model file paths
CNN_MODEL_PATH = MODELS_DIR / "product_quality_cnn.h5"
DNN_MODEL_PATH = MODELS_DIR / "price_prediction_dnn.h5"
CNN_ENCODER_PATH = MODELS_DIR / "product_encoder.pkl"
QUALITY_ENCODER_PATH = MODELS_DIR / "quality_encoder.pkl"
PRICE_SCALER_X_PATH = MODELS_DIR / "price_scaler_x.pkl"
PRICE_SCALER_Y_PATH = MODELS_DIR / "price_scaler_y.pkl"
PRICE_ENCODER_PATH = MODELS_DIR / "price_encoder.pkl"

# Global model variables
cnn_model = None
dnn_model = None
product_encoder = None
quality_encoder = None
price_scaler_x = None
price_scaler_y = None
price_encoder = None

# Constants
PRODUCT_TYPES = ["papa", "fresa", "arveja", "lechuga", "zanahoria", "cilantro", "tomate", "cebolla"]
QUALITY_LEVELS = ["Excelente", "Buena", "Estándar"]

# Fincas data (simulated)
FINCAS_TUNJA = {
    "finca_sol_naciente": {
        "nombre": "Finca Sol Naciente",
        "agricultor": "Don Pedro Pérez",
        "ubicacion": "Vereda El Porvenir, Tunja",
        "lat": 5.5398, "lon": -73.3421,
        "productos_cultivados": ["papa", "fresa", "arveja"]
    },
    "huerta_verde_boyaca": {
        "nombre": "Huerta Verde Boyacá",
        "agricultor": "Doña María Rojas",
        "ubicacion": "Vereda Runta, Tunja",
        "lat": 5.5123, "lon": -73.3678,
        "productos_cultivados": ["lechuga", "zanahoria", "cilantro"]
    },
    "finca_esperanza": {
        "nombre": "Finca La Esperanza",
        "agricultor": "Don Carlos Gómez",
        "ubicacion": "Vereda San Rafael, Tunja",
        "lat": 5.5501, "lon": -73.3512,
        "productos_cultivados": ["tomate", "cebolla", "papa"]
    }
}

# Pydantic Models
class ProductBase(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    nombre: str
    tipo_producto: str
    calidad_ia: str
    confianza_calidad_pct: float
    cantidad_kg: float
    precio_sugerido_cop: float
    precio_ia_activa: bool
    origen_finca: str
    agricultor: str
    ubicacion_finca_lat: float
    ubicacion_finca_lon: float
    fecha_publicacion: datetime = Field(default_factory=datetime.utcnow)
    blockchain_id: str
    disponible: bool = True

class CompraRequest(BaseModel):
    producto_id: str
    cantidad_kg: float
    metodo_entrega: str  # 'domicilio' o 'recogida_cultivo'
    comprador_info: dict

class AIStatus(BaseModel):
    cnn_activa: bool
    dnn_activa: bool
    modelos_entrenados: bool

# AI Training Functions
def generate_synthetic_image_data(num_samples=2000, img_size=(64, 64)):
    """Generate synthetic image data for CNN training"""
    try:
        data = []
        labels_product = []
        labels_quality = []
        
        logger.info(f"Generating {num_samples} synthetic image samples...")
        
        for _ in range(num_samples):
            product_type = random.choice(PRODUCT_TYPES)
            quality = random.choice(QUALITY_LEVELS)
            
            # Generate synthetic image data
            image = np.random.rand(img_size[0], img_size[1], 3) * 255
            
            # Add quality-based variations
            if quality == "Excelente":
                image = image * random.uniform(0.9, 1.0)  # Brighter
                image = np.clip(image + np.random.normal(0, 10, image.shape), 0, 255)
            elif quality == "Buena":
                image = image * random.uniform(0.7, 0.9)
                image = np.clip(image + np.random.normal(0, 20, image.shape), 0, 255)
            else:  # Estándar
                image = image * random.uniform(0.5, 0.7)
                image = np.clip(image + np.random.normal(0, 30, image.shape), 0, 255)
            
            image = image.astype(np.uint8)
            
            data.append(image)
            labels_product.append(product_type)
            labels_quality.append(quality)
        
        return pd.DataFrame({
            'image': data,
            'product_type': labels_product,
            'quality': labels_quality
        })
    except Exception as e:
        logger.error(f"Error generating synthetic image data: {e}")
        raise

def train_cnn_model():
    """Train CNN model for product quality assessment"""
    global cnn_model, product_encoder, quality_encoder
    
    try:
        logger.info("Training CNN model for product quality assessment...")
        
        # Generate synthetic data
        df_images = generate_synthetic_image_data(2500)
        
        # Prepare image data
        X_images = np.array(df_images['image'].tolist()) / 255.0  # Normalize
        
        # Encode labels
        product_encoder = OneHotEncoder(sparse_output=False)
        quality_encoder = OneHotEncoder(sparse_output=False)
        
        y_product = product_encoder.fit_transform(df_images[['product_type']])
        y_quality = quality_encoder.fit_transform(df_images[['quality']])
        
        # Split data
        X_train, X_test, y_prod_train, y_prod_test, y_qual_train, y_qual_test = train_test_split(
            X_images, y_product, y_quality, test_size=0.2, random_state=42
        )
        
        # Build CNN model
        cnn_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(len(PRODUCT_TYPES) + len(QUALITY_LEVELS), activation='softmax')
        ])
        
        cnn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Combine outputs for training
        y_combined_train = np.concatenate([y_prod_train, y_qual_train], axis=1)
        y_combined_test = np.concatenate([y_prod_test, y_qual_test], axis=1)
        
        # Train model
        history = cnn_model.fit(
            X_train, y_combined_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_combined_test),
            verbose=1
        )
        
        # Save model and encoders
        cnn_model.save(CNN_MODEL_PATH, save_format='h5')
        joblib.dump(product_encoder, CNN_ENCODER_PATH)
        joblib.dump(quality_encoder, QUALITY_ENCODER_PATH)
        
        logger.info("CNN model trained and saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error training CNN model: {e}")
        return False

def generate_synthetic_price_data(num_samples=3000):
    """Generate synthetic price data for DNN training"""
    try:
        data = []
        logger.info(f"Generating {num_samples} synthetic price samples...")
        
        for _ in range(num_samples):
            product_type = random.choice(PRODUCT_TYPES)
            quality = random.choice(QUALITY_LEVELS)
            
            # Simulate weather data for Tunja
            temp_avg = random.uniform(12, 25)  # Celsius
            humidity_avg = random.uniform(55, 95)  # Percentage
            day_of_year = random.randint(1, 365)
            
            # Base prices (COP per kg)
            base_prices = {
                "papa": 1800, "fresa": 6000, "arveja": 3500,
                "lechuga": 1200, "zanahoria": 1400, "cilantro": 900,
                "tomate": 2200, "cebolla": 1600
            }
            
            base_price = base_prices.get(product_type, 1500)
            
            # Quality adjustments
            quality_multipliers = {"Excelente": 1.3, "Buena": 1.1, "Estándar": 1.0}
            base_price *= quality_multipliers[quality]
            
            # Seasonal adjustments
            if 150 < day_of_year < 240:  # Summer months
                base_price *= random.uniform(1.05, 1.15)
            elif day_of_year < 60 or day_of_year > 330:  # Winter months
                base_price *= random.uniform(0.9, 1.05)
            
            # Weather impact
            if temp_avg > 22 and product_type in ["lechuga", "fresa"]:
                base_price *= random.uniform(1.1, 1.2)  # Heat stress
            if humidity_avg > 85:
                base_price *= random.uniform(0.95, 1.05)  # High humidity
            
            # Add random variation
            price_variation = random.uniform(0.85, 1.15)
            final_price = max(300, int(base_price * price_variation))
            
            data.append({
                'product_type': product_type,
                'quality': quality,
                'temp_avg': temp_avg,
                'humidity_avg': humidity_avg,
                'day_of_year': day_of_year,
                'price_per_kg': final_price
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Error generating synthetic price data: {e}")
        raise

def train_dnn_model():
    """Train DNN model for price prediction"""
    global dnn_model, price_scaler_x, price_scaler_y, price_encoder
    
    try:
        logger.info("Training DNN model for price prediction...")
        
        # Generate synthetic data
        df_prices = generate_synthetic_price_data(4000)
        
        # Prepare features
        X = df_prices[['product_type', 'quality', 'temp_avg', 'humidity_avg', 'day_of_year']].copy()
        y = df_prices[['price_per_kg']].copy()
        
        # Encode categorical variables
        price_encoder = OneHotEncoder(sparse_output=False)
        X_categorical = price_encoder.fit_transform(X[['product_type', 'quality']])
        X_numerical = X[['temp_avg', 'humidity_avg', 'day_of_year']].values
        X_processed = np.concatenate([X_numerical, X_categorical], axis=1)
        
        # Scale features
        price_scaler_x = MinMaxScaler()
        price_scaler_y = MinMaxScaler()
        
        X_scaled = price_scaler_x.fit_transform(X_processed)
        y_scaled = price_scaler_y.fit_transform(y.values)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        # Build DNN model
        dnn_model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])
        
        dnn_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = dnn_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save model and scalers
        dnn_model.save(DNN_MODEL_PATH, save_format='h5')
        joblib.dump(price_scaler_x, PRICE_SCALER_X_PATH)
        joblib.dump(price_scaler_y, PRICE_SCALER_Y_PATH)
        joblib.dump(price_encoder, PRICE_ENCODER_PATH)
        
        logger.info("DNN model trained and saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error training DNN model: {e}")
        return False

def load_ai_models():
    """Load trained AI models"""
    global cnn_model, dnn_model, product_encoder, quality_encoder
    global price_scaler_x, price_scaler_y, price_encoder
    
    try:
        # Load CNN model
        if CNN_MODEL_PATH.exists():
            cnn_model = load_model(CNN_MODEL_PATH)
            product_encoder = joblib.load(CNN_ENCODER_PATH)
            quality_encoder = joblib.load(QUALITY_ENCODER_PATH)
            logger.info("CNN model loaded successfully")
        
        # Load DNN model
        if DNN_MODEL_PATH.exists():
            dnn_model = load_model(DNN_MODEL_PATH)
            price_scaler_x = joblib.load(PRICE_SCALER_X_PATH)
            price_scaler_y = joblib.load(PRICE_SCALER_Y_PATH)
            price_encoder = joblib.load(PRICE_ENCODER_PATH)
            logger.info("DNN model loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading AI models: {e}")
        return False

async def predict_product_quality(image_bytes):
    """Predict product type and quality from image"""
    global cnn_model, product_encoder, quality_encoder
    
    try:
        if cnn_model is None:
            return {"error": "CNN model not loaded"}
        
        # Process image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = cnn_model.predict(img_array, verbose=0)
        
        # Split prediction (assuming concatenated output)
        n_products = len(PRODUCT_TYPES)
        product_probs = prediction[0][:n_products]
        quality_probs = prediction[0][n_products:]
        
        # Get predicted classes
        predicted_product_idx = np.argmax(product_probs)
        predicted_quality_idx = np.argmax(quality_probs)
        
        predicted_product = PRODUCT_TYPES[predicted_product_idx]
        predicted_quality = QUALITY_LEVELS[predicted_quality_idx]
        
        confidence_product = float(product_probs[predicted_product_idx])
        confidence_quality = float(quality_probs[predicted_quality_idx])
        
        return {
            "tipo_producto": predicted_product,
            "calidad_estimada": predicted_quality,
            "confianza_producto_pct": round(confidence_product * 100, 2),
            "confianza_calidad_pct": round(confidence_quality * 100, 2),
            "ia_activa": True
        }
        
    except Exception as e:
        logger.error(f"Error in product quality prediction: {e}")
        return {
            "tipo_producto": random.choice(PRODUCT_TYPES),
            "calidad_estimada": random.choice(QUALITY_LEVELS),
            "confianza_producto_pct": 0,
            "confianza_calidad_pct": 0,
            "ia_activa": False
        }

async def predict_fair_price(product_type, quality, temp_avg=18, humidity_avg=75):
    """Predict fair price using DNN model"""
    global dnn_model, price_scaler_x, price_scaler_y, price_encoder
    
    try:
        if dnn_model is None:
            # Fallback pricing
            base_prices = {
                "papa": 1800, "fresa": 6000, "arveja": 3500,
                "lechuga": 1200, "zanahoria": 1400, "cilantro": 900,
                "tomate": 2200, "cebolla": 1600
            }
            base_price = base_prices.get(product_type, 1500)
            quality_mult = {"Excelente": 1.3, "Buena": 1.1, "Estándar": 1.0}
            return {
                "precio_sugerido": int(base_price * quality_mult.get(quality, 1.0)),
                "ia_activa": False
            }
        
        # Prepare input data
        day_of_year = datetime.now().timetuple().tm_yday
        input_data = pd.DataFrame([[product_type, quality, temp_avg, humidity_avg, day_of_year]],
                                columns=['product_type', 'quality', 'temp_avg', 'humidity_avg', 'day_of_year'])
        
        # Encode categorical variables
        X_categorical = price_encoder.transform(input_data[['product_type', 'quality']])
        X_numerical = input_data[['temp_avg', 'humidity_avg', 'day_of_year']].values
        X_processed = np.concatenate([X_numerical, X_categorical], axis=1)
        
        # Scale input
        X_scaled = price_scaler_x.transform(X_processed)
        
        # Make prediction
        predicted_scaled = dnn_model.predict(X_scaled, verbose=0)
        predicted_price = price_scaler_y.inverse_transform(predicted_scaled)[0][0]
        
        return {
            "precio_sugerido": max(300, int(predicted_price)),
            "ia_activa": True
        }
        
    except Exception as e:
        logger.error(f"Error in price prediction: {e}")
        # Fallback
        base_prices = {
            "papa": 1800, "fresa": 6000, "arveja": 3500,
            "lechuga": 1200, "zanahoria": 1400, "cilantro": 900,
            "tomate": 2200, "cebolla": 1600
        }
        base_price = base_prices.get(product_type, 1500)
        return {"precio_sugerido": base_price, "ia_activa": False}

# API Routes
@api_router.get("/")
async def root():
    return {"message": "AgroDirecto Tunja API - Sistema de Agricultura Inteligente"}

@api_router.get("/productos", response_model=List[ProductBase])
async def get_productos():
    """Get all available products"""
    try:
        products = await db.productos.find({"disponible": True}).to_list(100)
        return [ProductBase(**product) for product in products]
    except Exception as e:
        logger.error(f"Error getting products: {e}")
        return []

@api_router.post("/publicar_producto")
async def publicar_producto(
    imagen: UploadFile = File(...),
    nombre: str = Form(None),
    cantidad_kg: float = Form(1.0)
):
    """Publish a new product with AI analysis"""
    try:
        # Read image
        image_bytes = await imagen.read()
        
        # AI Quality Analysis
        quality_result = await predict_product_quality(image_bytes)
        
        if not quality_result.get("ia_activa", False):
            logger.warning("AI quality analysis failed, using fallback")
        
        tipo_producto = quality_result["tipo_producto"]
        calidad_estimada = quality_result["calidad_estimada"]
        
        # AI Price Prediction
        price_result = await predict_fair_price(tipo_producto, calidad_estimada)
        
        # Get random farm info
        finca_key = random.choice(list(FINCAS_TUNJA.keys()))
        finca_info = FINCAS_TUNJA[finca_key]
        
        # Create product
        product = ProductBase(
            nombre=nombre or f"{tipo_producto.capitalize()} de Calidad {calidad_estimada}",
            tipo_producto=tipo_producto,
            calidad_ia=calidad_estimada,
            confianza_calidad_pct=quality_result["confianza_calidad_pct"],
            cantidad_kg=cantidad_kg,
            precio_sugerido_cop=price_result["precio_sugerido"],
            precio_ia_activa=price_result["ia_activa"],
            origen_finca=finca_info["nombre"],
            agricultor=finca_info["agricultor"],
            ubicacion_finca_lat=finca_info["lat"],
            ubicacion_finca_lon=finca_info["lon"],
            blockchain_id=f"AGR{random.randint(100000, 999999)}"
        )
        
        # Save to database
        result = await db.productos.insert_one(product.model_dump())
        
        logger.info(f"Product published: {product.nombre} - {product.id}")
        
        return {
            "estado": "exito",
            "mensaje": "Producto publicado exitosamente",
            "producto": product.model_dump()
        }
        
    except Exception as e:
        logger.error(f"Error publishing product: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@api_router.post("/comprar_producto")
async def comprar_producto(compra: CompraRequest):
    """Purchase a product"""
    try:
        # Find product
        product = await db.productos.find_one({"id": compra.producto_id, "disponible": True})
        
        if not product:
            raise HTTPException(status_code=404, detail="Producto no encontrado")
        
        if product["cantidad_kg"] < compra.cantidad_kg:
            raise HTTPException(status_code=400, detail="Cantidad insuficiente")
        
        # Calculate total price
        total_price = compra.cantidad_kg * product["precio_sugerido_cop"]
        
        # Update product quantity or remove if sold out
        new_quantity = product["cantidad_kg"] - compra.cantidad_kg
        if new_quantity <= 0:
            await db.productos.update_one(
                {"id": compra.producto_id},
                {"$set": {"disponible": False, "cantidad_kg": 0}}
            )
        else:
            await db.productos.update_one(
                {"id": compra.producto_id},
                {"$set": {"cantidad_kg": new_quantity}}
            )
        
        # Create purchase record
        purchase = {
            "id": str(uuid.uuid4()),
            "producto_id": compra.producto_id,
            "producto_nombre": product["nombre"],
            "cantidad_kg": compra.cantidad_kg,
            "precio_total": total_price,
            "metodo_entrega": compra.metodo_entrega,
            "comprador_info": compra.comprador_info,
            "agricultor": product["agricultor"],
            "fecha_compra": datetime.utcnow(),
            "blockchain_tx": f"TXN{random.randint(100000, 999999)}"
        }
        
        await db.compras.insert_one(purchase)
        
        logger.info(f"Purchase completed: {compra.producto_id} - {compra.cantidad_kg}kg")
        
        return {
            "estado": "exito",
            "mensaje": f"Compra realizada exitosamente",
            "compra": purchase,
            "total_cop": total_price
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing purchase: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@api_router.get("/status_ia", response_model=AIStatus)
async def get_ia_status():
    """Get AI models status"""
    return AIStatus(
        cnn_activa=cnn_model is not None,
        dnn_activa=dnn_model is not None,
        modelos_entrenados=(cnn_model is not None and dnn_model is not None)
    )

@api_router.post("/entrenar_modelos")
async def entrenar_modelos():
    """Train AI models (this might take a while)"""
    try:
        logger.info("Starting AI model training...")
        
        # Train models
        cnn_success = train_cnn_model()
        dnn_success = train_dnn_model()
        
        if cnn_success and dnn_success:
            # Load the newly trained models
            load_ai_models()
            return {
                "estado": "exito",
                "mensaje": "Modelos entrenados y cargados exitosamente"
            }
        else:
            return {
                "estado": "error",
                "mensaje": "Error durante el entrenamiento de modelos"
            }
            
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise HTTPException(status_code=500, detail=f"Error entrenando modelos: {str(e)}")

@api_router.get("/fincas")
async def get_fincas():
    """Get information about farms"""
    return {"fincas": FINCAS_TUNJA}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize AI models on startup"""
    logger.info("Starting AgroDirecto Tunja API...")
    
    # Try to load existing models
    models_loaded = load_ai_models()
    
    if not models_loaded:
        logger.info("No existing models found. Training new models...")
        # Train models in background
        asyncio.create_task(train_models_background())

async def train_models_background():
    """Train models in background"""
    try:
        cnn_success = train_cnn_model()
        dnn_success = train_dnn_model()
        
        if cnn_success and dnn_success:
            load_ai_models()
            logger.info("Background model training completed successfully")
        else:
            logger.error("Background model training failed")
    except Exception as e:
        logger.error(f"Error in background model training: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()