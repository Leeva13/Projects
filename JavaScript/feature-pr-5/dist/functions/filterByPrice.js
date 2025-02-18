export const filterByPrice = (products, maxPrice) => {
    return products.filter(product => product.price <= maxPrice);
};
