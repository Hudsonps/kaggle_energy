/// <reference path="jquery-1.4.1-vsdoc.js"/>
/// <reference path="data.js"/>
$(document).ready(function () {
    $("#btnM0").click(function () {
        doCharts(dataM0);
    });
    $("#btnM1").click(function () {
        doCharts(dataM1);
    });
    $("#btnM2").click(function () {
        doCharts(dataM2);
    });
    $("#btnM3").click(function () {
        doCharts(dataM3);
    });
});

// this function creates the plot, then creates the
// html code that is then saved under content
// Inspect the source of the plots in the browser to see this
function doCharts(alldata) {
    var content = $("#content");
    content.children().remove();
    for (var i = 0; i < alldata.length; i++) {
        var url = chartUrl(
            alldata[i]["data"],
            alldata[i]["b_id"]
        );
        // add </br> to place figure in next line
        content.append('' + '<img src="' + url + '" />');
    }
}

// !!! Need to replace this with something that doesnt depend on Google
// so there are no resource limits
// !!! Chart fails when data point is a nan (or not numeric)
// Interpolation must be done before data is saved in dataM#.js
function chartUrl(data, i) {
    var res = "https://chart.apis.google.com/chart?chs=300x150&chls=1&cht=lc&chtt=" + i + "&chd=";
    var maxval = Math.max.apply(Math, data);
    // console.log(i, maxval, data);
    return res + extendedEncode(data, maxval);
}

//// Under this line is Google's chart encoding functions, copied and pasted from their site at:
////   http://code.google.com/apis/chart/docs/data_formats.html#encoding_data

// Same as simple encoding, but for extended encoding.
var EXTENDED_MAP =
  'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-.';
var EXTENDED_MAP_LENGTH = EXTENDED_MAP.length;
function extendedEncode(arrVals, maxVal) {
    var chartData = 'e:';

    for (i = 0, len = arrVals.length; i < len; i++) {
        // In case the array vals were translated to strings.
        var numericVal = new Number(arrVals[i]);
        // Scale the value to maxVal.
        var scaledVal = Math.floor(EXTENDED_MAP_LENGTH *
        EXTENDED_MAP_LENGTH * numericVal / maxVal);

        if (scaledVal > (EXTENDED_MAP_LENGTH * EXTENDED_MAP_LENGTH) - 1) {
            chartData += "..";
        } else if (scaledVal < 0) {
            chartData += '__';
        } else {
            // Calculate first and second digits and add them to the output.
            var quotient = Math.floor(scaledVal / EXTENDED_MAP_LENGTH);
            var remainder = scaledVal - EXTENDED_MAP_LENGTH * quotient;
            chartData += EXTENDED_MAP.charAt(quotient) + EXTENDED_MAP.charAt(remainder);
        }
    }

    return chartData;
}