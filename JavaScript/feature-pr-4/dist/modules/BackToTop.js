export function initBackToTop() {
    $(window).scroll(function () {
        if ($(window).scrollTop() > 100) {
            $('.back-to-top').fadeIn('slow');
        }
        else {
            $('.back-to-top').fadeOut('slow');
        }
    });
    $('.back-to-top').click(function () {
        $('html, body').animate({ scrollTop: 0 }, 1500, 'easeInOutExpo');
        return false;
    });
}
//# sourceMappingURL=BackToTop.js.map