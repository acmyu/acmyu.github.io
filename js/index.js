/*!
 * Start Bootstrap - Agency Bootstrap Theme (http://startbootstrap.com)
 * Code licensed under the Apache License v2.0.
 * For details, see http://www.apache.org/licenses/LICENSE-2.0.
 */

// jQuery for page scrolling feature - requires jQuery Easing plugin
$(function() {
    $('a.page-scroll').bind('click', function(event) {
        var $anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: $($anchor.attr('href')).offset().top
        }, 1500, 'easeInOutExpo');
        event.preventDefault();
    });
});

// Highlight the top nav as scrolling occurs
$('body').scrollspy({
    target: '.navbar-fixed-top'
})

// Closes the Responsive Menu on Menu Item Click
$('.navbar-collapse ul li a').click(function() {
    $('.navbar-toggle:visible').click();
});

var fullScreenMode = $(window).width() >= 768;
$( document ).ready(function() {
	if(!fullScreenMode){
		$(".portfolio-item").slice(3).each(function() {
			$( this ).hide();
		});
		$("#more-projects").show();
	}
});


$(window).resize(function() {
	if ($(window).width() < 768) {
		if(fullScreenMode){
			$(".portfolio-item").slice(3).each(function() {
				$( this ).hide();
			});
			$("#more-projects").show();
		}
		fullScreenMode = false;
	}
	else {
		if(!fullScreenMode){
			$(".portfolio-item").each(function() {
				$( this ).show();
			});
			$("#more-projects").hide();
		}
		fullScreenMode = true;
	}
});

$("#show-more").click(function() {
	if(!fullScreenMode){
		$(".portfolio-item:hidden").slice(0,3).each(function() {
			$( this ).show();
		});
		if($(".portfolio-item:hidden").length==0) {
			$("#more-projects").hide();
		}
	}
});