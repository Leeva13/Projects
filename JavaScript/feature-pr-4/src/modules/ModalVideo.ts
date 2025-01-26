export function initModalVideo(): void {
    let videoSrc: string = "";

    $('.btn-play').click(function () {
        videoSrc = $(this).data("src") as string;
    });

    $('#videoModal').on('shown.bs.modal', function () {
        $("#video").attr('src', videoSrc + "?autoplay=1&amp;modestbranding=1&amp;showinfo=0");
    });

    $('#videoModal').on('hide.bs.modal', function () {
        $("#video").attr('src', videoSrc);
    });
}
export {};