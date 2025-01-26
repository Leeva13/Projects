export function initModalVideo() {
    let videoSrc = "";
    $('.btn-play').click(function () {
        videoSrc = $(this).data("src");
    });
    $('#videoModal').on('shown.bs.modal', function () {
        $("#video").attr('src', videoSrc + "?autoplay=1&amp;modestbranding=1&amp;showinfo=0");
    });
    $('#videoModal').on('hide.bs.modal', function () {
        $("#video").attr('src', videoSrc);
    });
}
//# sourceMappingURL=ModalVideo.js.map