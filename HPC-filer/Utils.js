var Utils = new Object();

//*********************************************************************************************************
// Adds an event handler to an event on a target.
//*********************************************************************************************************
Utils.AddEventHandler = function( target, handler, eventName )
{
	var eventName = window.addEventListener ? eventName : "on" + eventName;
	
	if( target.addEventListener )
	{
		target.addEventListener( eventName, handler, false );
	}
	else if( target.attachEvent )
	{
		target.attachEvent( eventName, handler );
	}
}


//*********************************************************************************************************
// Returns true if the array contains the given value.
//*********************************************************************************************************
Utils.ArrayContains = function( array, value )
{
	for( var index = 0; index < array.length; index++ )
	{
		if( array[index] == value )
		{
			return true;
		}
	}
	
	return false;
}


//*********************************************************************************************************
// Gets the value of the given cookie.
//*********************************************************************************************************
Utils.GetCookie = function( cookieName )
{
	var regEx = new RegExp( cookieName + "=[^;]+", "i" );
	if( document.cookie.match( regEx ) )
	{
		return document.cookie.match( regEx )[0].split( "=" )[1];
	}
	
	return "";
}


//*********************************************************************************************************
// Returns true if the given object is undefined.
//*********************************************************************************************************
Utils.IsUndefined = function( value )
{
	return typeof( value ) == "undefined";
}


//*********************************************************************************************************
// Sets the value of the given cookie.
//*********************************************************************************************************
Utils.SetCookie = function( cookieName, value, days )
{
	var expireDate = new Date();
	expireDate.setDate( expireDate.getDate() + parseInt( days ) );
	document.cookie = cookieName + "=" + value + "; expires=" + expireDate.toGMTString() + "; path=/";
}


//*********************************************************************************************************
// Prevents further propagation of the current event.
//*********************************************************************************************************
Utils.StopPropagation = function( e )
{
	if( !Utils.IsUndefined( e ) )
	{
		e.stopPropagation();
	}
	else
	{
		event.cancelBubble = true;
	}
}
