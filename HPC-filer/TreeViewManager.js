var PersistedData = new Object();

var TreeViewManager = new Object();
TreeViewManager.closeImagePath = "/images/universal_menu/closed_lite.gif"
TreeViewManager.openImagePath = "/images/universal_menu/open_lite.gif"
TreeViewManager.branchNodeName = "ul";

//*********************************************************************************************************
// Registers a UL tag as being your tree view.
//*********************************************************************************************************
TreeViewManager.RegisterTree = function( treeID, enablePersist, persistDays )
{
	var tree = document.getElementById( treeID );
	if( Utils.IsUndefined( tree ) )
	{
		alert( 'Tree \'' + treeID + '\' was not found.' );
		return;
	}
	
	var ulElements = tree.getElementsByTagName( TreeViewManager.branchNodeName );
	if( Utils.IsUndefined( ulElements ) || ulElements.length == 0 )
	{
		return;
	}
	
	if( Utils.IsUndefined( PersistedData[treeID] ) )
	{
		PersistedData[treeID] = enablePersist && ( Utils.GetCookie( treeID ) != "" ) ? Utils.GetCookie( treeID ).split( "," ) : "";
	}
	
	for( var index = 0; index < ulElements.length; index++ )
	{
		TreeViewManager.BuildBranch( treeID, ulElements[index], index );
	}
	
	if( enablePersist )
	{
		var durationDays = Utils.IsUndefined( persistDays ) ? 1 : parseInt( persistDays );
		Utils.AddEventHandler( window, function() { TreeViewManager.RememberState( treeID, durationDays ) }, "unload");
	}
}

//*********************************************************************************************************
// Builds the branches of the tree view.
//*********************************************************************************************************
TreeViewManager.BuildBranch = function( treeID, ulElement, index )
{
	ulElement.parentNode.className = "submenu";
	
	if( typeof( PersistedData[treeID] ) == "object" )
	{
		if( Utils.ArrayContains( PersistedData[treeID], index ) )
		{
			ulElement.setAttribute( "rel", "open" );
			ulElement.style.display = "block";
			ulElement.parentNode.style.backgroundImage = "url(" + TreeViewManager.openImagePath + ")";
		}
		else
		{
			ulElement.setAttribute( "rel", "closed" );
			ulElement.style.display = "none";
			ulElement.parentNode.style.backgroundImage = "url(" + TreeViewManager.closeImagePath + ")";
		}
	}
	else if( ulElement.getAttribute( "rel" ) == null || ulElement.getAttribute( "rel" ) == false )
	{
		ulElement.setAttribute( "rel", "closed" );
	}
	else if( ulElement.getAttribute( "rel" ) == "open" )
	{
		TreeViewManager.ExpandNode( treeID, ulElement );
	}
	
	ulElement.parentNode.onclick = function( e )
	{
		var subMenu = this.getElementsByTagName( TreeViewManager.branchNodeName )[0];
		if( subMenu.getAttribute( "rel" ) == "closed" )
		{
			subMenu.style.display = "block";
			subMenu.setAttribute( "rel", "open" );
			ulElement.parentNode.style.backgroundImage = "url(" + TreeViewManager.openImagePath + ")";
		}
		else if( subMenu.getAttribute( "rel" ) == "open" )
		{
			subMenu.style.display = "none";
			subMenu.setAttribute( "rel", "closed" );
			ulElement.parentNode.style.backgroundImage = "url(" + TreeViewManager.closeImagePath + ")";
		}
		Utils.StopPropagation( e );
	}

	ulElement.onclick = function( e )
	{
		Utils.StopPropagation( e );
	}
}

//*********************************************************************************************************
// Expands the given tree view node.
//*********************************************************************************************************
TreeViewManager.ExpandNode = function( treeID, ulElement )
{
	var rootNode = document.getElementById( treeID );
	var currentNode = ulElement;
	
	currentNode.style.display = "block";
	currentNode.parentNode.style.backgroundImage = "url(" + TreeViewManager.openImagePath + ")";
	
	while( currentNode != rootNode )
	{
		if( currentNode.tagName == TreeViewManager.branchNodeName.toUpper() )
		{
			currentNode.style.display = "block";
			currentNode.setAttribute( "rel", "open" );
			currentNode.parentNode.style.backgroundImage = "url(" + TreeViewManager.openImagePath + ")";
		}
		
		currentNode = currentNode.parentNode;
	}
}

//*********************************************************************************************************
// Set all tree view nodes to either being expanded or contracted.
//*********************************************************************************************************
TreeViewManager.SetAllNodes = function( treeID, action )
{
	var ulElements = document.getElementById( treeID ).getElementsByTagName( TreeViewManager.branchNodeName );
	for( var index = 0; index < ulElements.length; index++ )
	{
		ulElements[index].style.display = action == "expand" ? "block" : "none";
		var relvalue = action == "expand" ? "open" : "closed";
		ulElements[index].setAttribute( "rel", relvalue );
		ulElements[index].parentNode.style.backgroundImage = action == "expand" ? "url(" + TreeViewManager.openImagePath + ")" : "url(" + TreeViewManager.closeImagePath + ")";
	}
}

//*********************************************************************************************************
// Expands all tree view nodes in the given tree view.
//*********************************************************************************************************
TreeViewManager.OpenAll = function( treeID )
{
	TreeViewManager.SetAllNodes( treeID, "expand" );
}

//*********************************************************************************************************
// Contracts all tree view nodes in the given tree view.
//*********************************************************************************************************
TreeViewManager.CloseAll = function( treeID )
{
	TreeViewManager.SetAllNodes( treeID, "contract" );
}

//*********************************************************************************************************
// Stores the state of the given tree view.
//*********************************************************************************************************
TreeViewManager.RememberState = function( treeID, durationDays )
{
	var ulElements = document.getElementById( treeID ).getElementsByTagName( TreeViewManager.branchNodeName );
	var openuls = new Array();
	
	for( var index = 0; index < ulElements.length; index++ )
	{
		if( ulElements[index].getAttribute( "rel" ) == "open" )
		{
			openuls[openuls.length] = index;
		}
	}
	
	if( openuls.length == 0 )
	{
		openuls[0] = "none open";
	}

	Utils.SetCookie( treeID, openuls.join( "," ), durationDays );
}
