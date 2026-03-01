import React from 'react';
import { useCart } from '../context/CartContext';
import { Minus, Plus } from 'lucide-react';
import { getFoodImage } from '../utils/ImageHelper';

const MenuItem = ({ item, restaurantId }) => {
  const { cart, addToCart, removeFromCart } = useCart();

  // FIX: Force string comparison to ensure the counter doesn't go missing
  // This handles cases where one ID is a Number and the other is a String
  const cartItem = cart.find((i) => String(i.item_id) === String(item.item_id));
  const qty = cartItem ? cartItem.qty : 0;

  return (
    <div className="flex justify-between items-center py-6 border-b border-gray-100 bg-white group">
      <div className="flex-1 pr-4 text-left">
        {/* Veg/Non-Veg Indicator */}
        <div className={`w-3.5 h-3.5 border-2 ${item.is_veg ? 'border-green-600' : 'border-red-600'} flex items-center justify-center p-[2px] mb-1.5 rounded-sm`}>
          <div className={`w-full h-full rounded-full ${item.is_veg ? 'bg-green-600' : 'bg-red-600'}`}></div>
        </div>

        <h4 className="font-black text-gray-800 text-base leading-tight">{item.name}</h4>
        <p className="text-sm font-black text-gray-900 mt-0.5">₹{item.price}</p>

        <p className="text-[11px] text-gray-400 mt-2 line-clamp-2 leading-relaxed font-medium">
          {item.description?.includes('|') ? item.description.split('|').pop().trim() : item.description}
        </p>
      </div>

      <div className="relative flex-shrink-0">
        <div className="w-28 h-28 bg-gray-50 rounded-2xl overflow-hidden border border-gray-100 shadow-sm">
          <img
            src={getFoodImage(item.name)}
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
            alt={item.name}
            loading="lazy"
            onError={(e) => { e.target.src = "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&w=150&q=80"; }}
          />
        </div>

        {/* --- DYNAMIC TOGGLE LOGIC --- */}
        <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 shadow-xl rounded-xl overflow-hidden ring-4 ring-white min-w-[100px] bg-white">
          {qty === 0 ? (
            /* Show ADD Button if qty is 0 */
            <button
              onClick={() => addToCart(item, restaurantId)}
              className="w-full bg-white text-green-600 font-black py-2.5 text-xs uppercase transition-all active:scale-95"
            >
              Add
            </button>
          ) : (
            /* Show - QTY + Counter if qty > 0 */
            <div className="flex items-center justify-between py-2 px-2 text-green-600">
              <button
                onClick={() => removeFromCart(item.item_id)}
                className="p-1 hover:bg-gray-50 rounded-md active:scale-125 transition-transform"
              >
                <Minus size={16} strokeWidth={3} />
              </button>

              <span className="text-sm font-black text-gray-800 px-2">{qty}</span>

              <button
                onClick={() => addToCart(item, restaurantId)}
                className="p-1 hover:bg-gray-50 rounded-md active:scale-125 transition-transform"
              >
                <Plus size={16} strokeWidth={3} />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MenuItem;