export function initStickyNavbar() {
    $(window).scroll(function () {
        if ($(window).scrollTop() > 40) {
            $('.navbar').addClass('sticky-top');
        }
        else {
            $('.navbar').removeClass('sticky-top');
        }
    });
}
//# sourceMappingURL=StickyNavbar.js.map