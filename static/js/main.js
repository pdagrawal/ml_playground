// fade out message alerts
function fade_alerts() {
    alerts = document.getElementsByClassName("alert msg fade");
        var i = alerts.length;
        for (let elem of alerts) {
            i--;
            time = 3250+(1000*i);
            setTimeout(function() {
                $(elem).fadeOut("slow");
            }, time);
        }
}

// call fade out after DOMContentLoaded
window.addEventListener('DOMContentLoaded', (event) => {
    fade_alerts();
});