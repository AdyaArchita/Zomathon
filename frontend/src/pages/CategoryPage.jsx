import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ChevronLeft, Plus, Minus, Info, ChevronRight } from 'lucide-react'; // 1. Import Minus
import { getFoodImage } from '../utils/ImageHelper';
import { useCart } from '../context/CartContext';

const CategoryPage = () => {
  const { categoryName } = useParams();
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  // Destructure state and functions from Context
  const { cart, addToCart, removeFromCart, totalAmount, totalQty } = useCart();

  useEffect(() => {
    setLoading(true);
    fetch(`http://localhost:8000/category/${categoryName}`)
      .then(res => res.json())
      .then(data => {
        setItems(data);
        setLoading(false);
      })
      .catch(() => {
        setItems([]);
        setLoading(false);
      });
  }, [categoryName]);

  return (
    <div className="bg-white min-h-screen pb-40 font-sans">
      {/* Sticky Header */}
      <div className="sticky top-0 bg-white shadow-sm z-30 p-4 flex items-center border-b border-gray-50">
        <button onClick={() => navigate(-1)} className="p-1 hover:bg-gray-100 rounded-full transition-colors">
          <ChevronLeft size={24} className="text-gray-800" />
        </button>
        <div className="ml-3 text-left">
          <h1 className="text-lg font-black text-gray-800 uppercase tracking-tight">{categoryName}</h1>
          <p className="text-[10px] text-gray-400 font-bold uppercase tracking-widest">{items.length} Options Near You</p>
        </div>
      </div>

      <div className="p-4 space-y-12">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-40">
            <div className="w-10 h-10 border-4 border-red-500 border-t-transparent rounded-full animate-spin mb-4" />
            <p className="font-bold text-gray-300 uppercase text-xs tracking-widest">Searching Dataset...</p>
          </div>
        ) : items.length > 0 ? (
          items.map((item) => {
            // 2. Lookup logic: Find if item is in cart (using string-safe comparison)
            const cartItem = cart.find((i) => String(i.item_id) === String(item.item_id));
            const quantity = cartItem ? cartItem.qty : 0;

            return (
              <div key={item.item_id} className="flex justify-between items-start pb-10 border-b border-gray-100 last:border-0 relative group">

                {/* LEFT SIDE: Content */}
                <div className="flex-1 pr-4 text-left">
                  <div className="flex items-start space-x-2 mb-1.5">
                    <div className={`w-4 h-4 border-2 flex items-center justify-center rounded-sm mt-1 flex-shrink-0 ${item.is_veg ? 'border-green-600' : 'border-red-600'}`}>
                      <div className={`w-2 h-2 rounded-full ${item.is_veg ? 'bg-green-600' : 'bg-red-600'}`} />
                    </div>
                    <h3 className="font-black text-gray-800 text-lg leading-tight">{item.name}</h3>
                  </div>

                  <div className="ml-6">
                    <div className="flex items-center space-x-2">
                      <p className="text-sm font-black text-gray-900">₹{item.price}</p>
                      {item.price > 350 && (
                        <span className="text-[9px] font-black text-white bg-orange-400 px-1.5 py-0.5 rounded uppercase tracking-tighter">Bestseller</span>
                      )}
                    </div>
                    <p className="text-xs text-gray-400 mt-2 line-clamp-2 leading-relaxed font-medium">
                      {item.description.includes('|') ? item.description.split('|').pop().trim() : item.description}
                    </p>
                    <div className="mt-4 inline-flex items-center px-2 py-1 bg-gray-50 rounded-lg border border-gray-100">
                      <span className="text-[9px] text-gray-400 font-black uppercase tracking-widest leading-none">{item.restaurant_name}</span>
                    </div>
                  </div>
                </div>

                {/* RIGHT SIDE: Visuals */}
                <div className="relative flex-shrink-0">
                  <div className="w-36 h-36 rounded-3xl overflow-hidden shadow-2xl border border-gray-100 bg-gray-50">
                    <img
                      src={getFoodImage(item.name)}
                      alt={item.name}
                      className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                      loading="lazy"
                      onError={(e) => { e.target.src = "https://via.placeholder.com/300?text=Food+Image"; }}
                    />
                  </div>

                  {/* 3. DYNAMIC BUTTON TOGGLE */}
                  <div className="absolute -bottom-3 left-1/2 -translate-x-1/2 shadow-xl rounded-xl overflow-hidden ring-4 ring-white min-w-[100px] bg-white">
                    {quantity === 0 ? (
                      <button
                        onClick={() => addToCart(item, item.restaurant_id)}
                        className="w-full bg-white text-green-600 font-black py-2.5 text-xs uppercase transition-all active:scale-95 flex items-center justify-center px-6"
                      >
                        ADD <Plus size={14} className="ml-1 font-bold" />
                      </button>
                    ) : (
                      <div className="flex items-center justify-between py-2 px-2 text-green-600">
                        <button
                          onClick={() => removeFromCart(item.item_id)}
                          className="p-1 hover:bg-gray-50 rounded-md active:scale-125 transition-transform"
                        >
                          <Minus size={16} strokeWidth={3} />
                        </button>

                        <span className="text-sm font-black text-gray-800 px-2">{quantity}</span>

                        <button
                          onClick={() => addToCart(item, item.restaurant_id)}
                          className="p-1 hover:bg-gray-100 rounded-md active:scale-125 transition-transform"
                        >
                          <Plus size={16} strokeWidth={3} />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        ) : (
          <div className="flex flex-col items-center justify-center py-20 opacity-30">
            <Info size={48} className="mb-4" />
            <p className="font-black uppercase text-xs tracking-widest text-center">No {categoryName} found</p>
          </div>
        )}
      </div>

      {/* Sticky Cart Bar */}
      {totalQty > 0 && (
        <div className="fixed bottom-24 left-4 right-4 bg-[#E23744] text-white p-4 rounded-2xl flex justify-between items-center shadow-2xl z-50 animate-slide-up">
          <div className="flex flex-col text-left">
            <span className="text-[10px] font-black uppercase tracking-widest opacity-90">
              {totalQty} ITEM{totalQty > 1 ? 'S' : ''} ADDED
            </span>
            <span className="text-lg font-black tracking-tighter">₹{totalAmount}</span>
          </div>
          <button onClick={() => navigate('/order')} className="flex items-center font-black text-sm uppercase tracking-tighter">
            Next <ChevronRight size={18} className="ml-1" />
          </button>
        </div>
      )}
    </div>
  );
};

export default CategoryPage;