import { initStickyNavbar } from './modules/StickyNavbar';
import { initDropdownHover } from './modules/DropdownHover';
import { initModalVideo } from './modules/ModalVideo';
import { initBackToTop } from './modules/BackToTop';
import { initCounter } from './modules/Counter';
import { initTestimonialCarousel } from './modules/TestimonialCarousel';

(function ($: JQueryStatic) {
    "use strict";
    
    initStickyNavbar();
    initDropdownHover();
    initModalVideo();
    initBackToTop();
    initCounter();
    initTestimonialCarousel();

})(jQuery);
